import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from models import metric
from utils import *
from models.proo_head import PN_head
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter


use_gpu = torch.cuda.is_available()

import torch.nn.functional as F


from scipy.stats import t
import scipy

def get_cos_sin( x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val


def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def box_cox(data, beta):
    data_min = data.min()
    data = data - data_min
    if beta > 0:
        return (np.power(data[:, ], beta))/ beta
    else:
        return np.log(data)

def get_linear_clf(metric_type, feature_dimension, num_classes, scale_factor=None, margin=None):
    if metric_type == 'softmax':
        classifier = nn.Linear(feature_dimension, num_classes)
    elif metric_type == 'new':
        classifier = metric.CosineSimilarity_miuns(feature_dimension, num_classes, scale_factor=scale_factor)
    elif metric_type == 'cosine':
        classifier = metric.CosineSimilarity(feature_dimension, num_classes, scale_factor=scale_factor)
    elif metric_type == 'cosineface':
        classifier = metric.AddMarginProduct(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    elif metric_type == 'neg-softmax':
        classifier = metric.SoftmaxMargin(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    else:
        raise ValueError(f'Unknown metric type: "{metric_type}"')
    return classifier

def distribution_calibration(query, approximation, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(approximation)):
        dist.append(np.linalg.norm(query-approximation[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(approximation)[index], query[np.newaxis, :]])
    calibrated_mean = np.mean(mean, axis=0)
    calibrated_cov = np.mean(np.array(base_cov)[index], axis=0)+alpha

    return calibrated_mean, calibrated_cov

def draw_distribution(support_data, sampled_data, query_data, support_label, sampled_label, query_label, n_lsamples, n_usamples, save_name):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(n_components=2)

    tsne_input = np.concatenate((support_data, query_data, sampled_data))  # shape(80, 640)
    tsne_output = tsne.fit_transform(tsne_input)


    support_data_tsne = tsne_output[:n_lsamples, :]
    query_data_tsne = tsne_output[n_lsamples: n_lsamples+n_usamples, :]
    sample_data_tsne = tsne_output[n_lsamples+n_usamples:, :]

    plt.scatter(sample_data_tsne[:, 0], sample_data_tsne[:, 1], s=10, c=sampled_label, marker='x')
    plt.scatter(support_data_tsne[:, 0], support_data_tsne[:, 1], s=100, c=support_label, marker='*')
    plt.scatter(query_data_tsne[:, 0], query_data_tsne[:, 1], s=10, c=query_label, marker='v')
    plt.savefig('tsne/' + save_name + '.jpg')
    plt.close()



class SimpleHDF5Dataset:
    def __init__(self, file_handle = None):
        if file_handle == None:
            self.f = ''
            self.all_feats_dset = []
            self.all_labels = []
            self.total = 0
        else:
            self.f = file_handle
            self.all_feats_dset = self.f['all_feats'][...]
            self.all_labels = self.f['all_labels'][...]
            self.total = self.f['count'][0]
        # print('here')
    def __getitem__(self, i):
        return torch.Tensor(self.all_feats_dset[i,:]), int(self.all_labels[i])

    def __len__(self):
        return self.total

def evaluate(args, n_runs, ndatas, labels, classes, n_lsamples, n_ways, n_shot ,dataset_name, k):

    # ---- classification for each task

    acc_list = []
    acc_dis = {}

    print('Start classification for %d tasks...' % (n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform DC(2021 ICLR paper)
        beta = 0.7 # 0.8
        support_data = np.power(support_data[:, ], beta)
        query_data = np.power(query_data[:, ], beta)
        # support_data = box_cox(support_data,beta)
        # query_data = box_cox(query_data, beta)

        _, dim = support_data.shape

        X_aug = support_data
        Y_aug = support_label

        if args.appro_stastic == 'base_appro':
            features_list = []
            base_features_path = "./checkpoints/%s/WideResNet28_10_S2M2_R/base_features.plk"% dataset_name
            with open(base_features_path, 'rb') as f:
                data = pickle.load(f)
                for key in data.keys():
                    feature = np.array(data[key])
                    features_list.append([feature])
            feature = np.concatenate(np.array(features_list)).reshape(-1, dim)

        cls_name = args.cls

        # get approximated task centorid
        if args.appro_stastic == 'support':
            support_tensor = torch.Tensor(support_data)
            support_mean = torch.mean(support_tensor, dim=0, keepdim=True)
            approximation = F.normalize(support_mean).cuda()
        if args.appro_stastic == 'transductive':
            test_all = np.concatenate([support_data, query_data],axis=0)
            test_all = torch.Tensor(test_all).cuda()
            test_all = torch.mean(test_all, dim=0).unsqueeze(0)
            approximation = F.normalize(test_all).cuda()
        if args.appro_stastic == 'base_appro':
            feature = torch.Tensor(feature).cuda()
            feature = F.normalize(feature)
            support_tensor = torch.Tensor(support_data)
            support_mean = torch.mean(support_tensor, dim=0).unsqueeze(0)
            similar = torch.mm(F.normalize(support_mean).cuda(), F.normalize(feature).cuda().transpose(0,1))
            sim_cos, pred = similar[0].topk(k, 0, True, True)
            sim_weight = torch.pow(sim_cos, 0.5)/torch.sum(torch.pow(sim_cos, 0.5))
            approximation = torch.sum(sim_weight.unsqueeze(1)*feature[pred,:], dim=0).unsqueeze(0)
            approximation = F.normalize(approximation)

        # ---- classification
        linear_clf = get_linear_clf(cls_name, dim, n_ways, margin=0.0, scale_factor=5).cuda()
        finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        loss_function = nn.CrossEntropyLoss().cuda()
        batch_size = 4
        support_size = X_aug.shape[0]
        X_aug = torch.Tensor(X_aug).cuda()
        Y_aug = torch.LongTensor(Y_aug).cuda()
        query_data = torch.Tensor(query_data).cuda()
        query_label = torch.LongTensor(query_label).cuda()

        # ---- calculate the distance of support data to the task centroid
        if args.draw_selected_classes:
            from models.proo_head import L2SquareDist
            dis = L2SquareDist(X_aug.unsqueeze(0), approximation.unsqueeze(0))

        # ---- train classifier
        for _ in range(100):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):

                selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                z_batch = X_aug[selected_id]
                y_batch = Y_aug[selected_id]
                if cls_name == 'new':
                    scores= linear_clf(z_batch, approximation)
                else:
                    scores = linear_clf(z_batch)
                loss = loss_function(scores, y_batch)

                finetune_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                finetune_optimizer.step()

        if cls_name == 'new':
            scores= linear_clf(query_data, approximation)
        else:
            scores = linear_clf(query_data)

        # ---- calculate the accuracy
        acc = accuracy(scores, query_label).detach().cpu().numpy()
        acc_list.append(acc)

        # ---- save the accuracy with the distance to the task centroid
        if args.draw_selected_classes:
            dis = dis.cpu().numpy()
            dis = dis.mean()
            dis = np.around(dis, 1)
            if dis in acc_dis.keys():
                acc_dis[dis].append(acc)
            else:
                acc_dis[dis] = [acc]

    print('dataset %s,  %d way %d shot, k=%d, cls is %s'%(dataset_name, n_ways, n_shot, k, cls_name), ' ACC is: ',  mean_confidence_interval(acc_list))

    if args.draw_selected_classes:
        for key in acc_dis:
            print('dis is:', key, 'acc is:', np.array(acc_dis[key]).mean())
        with open('./pickle_file/acc_dis_'+ args.cls +'_'+str(select_class), 'wb') as f:
            pickle.dump(acc_dis, f)


def main_train(args, select_class=None):

    dataset_name = 'miniImagenet'    #'miniImagenet or tiered_imagenet'
    n_shot = args.n_shot
    n_ways = args.n_ways
    n_queries = 15


    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}

    FSLTask.loadDataSet('miniImagenet')



    FSLTask.setRandomStates(cfg)
    ndatas, classes = FSLTask.GenerateRunSet(end=args.n_runs, cfg=cfg, select_class=select_class)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(args.n_runs, n_samples, -1)  # shape [10000, 80, 640]
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(args.n_runs, n_shot + n_queries, n_ways).clone().view(args.n_runs,n_samples)

    evaluate(args, args.n_runs, ndatas, labels, classes, n_lsamples=n_lsamples, n_ways=n_ways, n_shot=n_shot,  dataset_name=dataset_name, k=args.num_neighbors)

if __name__ == '__main__':
    # ---- data loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--appro_stastic', type=str, default='base_appro',
                        help='support/transductive/base_appro')
    parser.add_argument('--num_neighbors', type=int, default='15000',
                        help='20000/15000/10000/5000/1000/500')
    parser.add_argument('--cls' , type=str, default='new', help='cosine/new')
    parser.add_argument('--n_shot' , type=int, default=1, help='1/5')
    parser.add_argument('--n_ways' , type=int, default=5, help='2/5/10')
    parser.add_argument('--draw_selected_classes', type=bool, default=True)
    parser.add_argument('--n_runs', type=int, default=2000)



    args = parser.parse_args()
    if args.draw_selected_classes:
        args.n_ways = 2
        select_class =[2,3] # the 2-th and 9-th classes in novel classes
        args.n_runs = 10000
    else:
        select_class = None

    main_train(args, select_class)


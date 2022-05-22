import pickle
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data.dataset import SimpleDataset_VAE, CUBDateset_VAE, tieredImageNet
from models.vae import LinearVAE
from models import metric
from utils import *
from models.proo_head import PN_head
from sklearn.manifold import TSNE
from torch.utils.tensorboard import SummaryWriter
from data_analysis import *
from torch.nn import init

use_gpu = torch.cuda.is_available()

BEST_ACC = 0
import torch.nn.functional as F

def cross_weight_cross_entropy(x_input, y_target, weight):
    softmax_func=nn.Softmax(dim=1)
    soft_output=softmax_func(x_input)
    # print('soft_output:\n',soft_output)
    b = x_input.shape[0]

    #在softmax的基础上取log
    log_output=torch.log(soft_output)
    # print('log_output:\n',log_output)

    #对比softmax与log的结合与nn.LogSoftmaxloss(负对数似然损失)的输出结果，发现两者是一致的。
    logsoftmax_func=nn.LogSoftmax(dim=1)
    logsoftmax_output=logsoftmax_func(x_input) * weight.view(b,1)
    # print('logsoftmax_output:\n',logsoftmax_output)

    #pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
    nllloss_func=nn.NLLLoss()
    nlloss_output=nllloss_func(logsoftmax_output,y_target)
    # print('nlloss_output:\n',nlloss_output)

    return nlloss_output

def weigth_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        nn.init.constant_(m.bias, 0.1)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.item(), 1)
        nn.init.constant_(m.bias.item(), 0)


        # logging.basicConfig(filename='var0.21_epoch_181_sample_3.txt', level=print)


from scipy.stats import t
import scipy

def get_cos_sin( x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val

def get_cos_sin_2( x, y):
    cos_val = torch.mm(x, y.t())/ torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val

def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def get_linear_clf(metric_type, feature_dimension, num_classes, scale_factor=None, margin=None, beta=1, concat=False):
    if metric_type == 'softmax':
        classifier = nn.Linear(feature_dimension, num_classes)
    elif metric_type == 'new':
        classifier = metric.CosineSimilarity_miuns(feature_dimension, num_classes, scale_factor=scale_factor, beta=beta, concat=args.concat)
    elif metric_type == 'cosine':
        classifier = metric.CosineSimilarity(feature_dimension, num_classes, scale_factor=scale_factor)
    elif metric_type == 'cosineface':
        classifier = metric.AddMarginProduct(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    elif metric_type == 'neg-softmax':
        classifier = metric.SoftmaxMargin(feature_dimension, num_classes, scale_factor=scale_factor, margin=margin)
    else:
        raise ValueError(f'Unknown metric type: "{metric_type}"')
    return classifier

def distribution_calibration(query, base_means, base_cov, k, alpha=0.21):
    dist = []
    for i in range(len(base_means)):
        dist.append(np.linalg.norm(query-base_means[i]))
    index = np.argpartition(dist, k)[:k]
    mean = np.concatenate([np.array(base_means)[index], query[np.newaxis, :]])
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

def evaluate(args, linear_vae, linear_vae_base, n_runs, ndatas, labels, n_lsamples, n_usamples, num_sampled, n_ways, n_shot, writer, step, dataset_name, k, cls_beta):

    # ---- classification for each task
    dif_sampled_list = []
    dif_original_list_1 = []
    dif_original_list_2 = []

    acc_list = []
    acc_dis = {}
    print('Start classification for %d tasks...' % (n_runs))
    for i in tqdm(range(n_runs)):

        support_data = ndatas[i][:n_lsamples].numpy()
        support_label = labels[i][:n_lsamples].numpy()
        query_data = ndatas[i][n_lsamples:].numpy()
        query_label = labels[i][n_lsamples:].numpy()
        # ---- Tukey's transform
        beta = 0.7 # 0.8
        support_data = np.power(support_data[:, ], beta)
        query_data = np.power(query_data[:, ], beta)

        _, dim = support_data.shape

        X_aug = support_data
        Y_aug = support_label

        base_means = []
        base_cov = []
        features_list = []
        base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/%s/WideResNet28_10_S2M2_R/base_features.plk"% dataset_name
        # base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_9s/base_features_new.plk"
        # base_features_path = '/data/jingxu/project/fsl_inv_equ/save_features/ResNet12_in_eq_distill/base_features.plk'
        # base_features_path = '/data/jingxu/project/fsl_distribution/checkpoints/tiered_imagenet/WideResNet28_10_S2M2_R/last/base_features.plk'
        # base_features_path = '/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_zscore/base_features.plk'
        with open(base_features_path, 'rb') as f:
            data = pickle.load(f)
            index = 0
            for key in data.keys():
                feature = np.array(data[key])
                # if index == 0:
                #     index = 1
                #     feature = feature_now
                # print(feature.shape)
                # feature = np.concatenate([feature, feature_now], axis=0)
                features_list.append([feature])
                mean = np.mean(feature, axis=0)
                cov = np.cov(feature.T)
                base_means.append(mean)
                base_cov.append(cov)

        cls_name = args.cls
        cls_num = k

        # # use baseline ++  base classes
        # import h5py
        # base_features_path = '/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.9_aug/base.hdf5'
        # with h5py.File(base_features_path, 'r') as f:
        #     fileset = SimpleHDF5Dataset(f)
        # feats = fileset.all_feats_dset
        # labels_ = fileset.all_labels
        # while np.sum(feats[-1]) == 0:
        #     feats  = np.delete(feats,-1,axis = 0)
        #     labels = np.delete(labels_,-1,axis = 0)
        # dataset = dict()
        # dataset['data'] = torch.FloatTensor(feats)
        # dataset['labels'] = torch.LongTensor(labels_)
        #
        # # base_means = torch.mean(dataset['data'] , dim=0).unsqueeze(0).cuda()
        # # base_means = F.normalize(base_means)
        # base_means = dataset['data'].numpy()
        # feature = dataset['data']

        # draw hotmap
        # base_means = torch.Tensor(base_means).cuda()
        # base_means = torch.mean(base_means, dim=0).unsqueeze(0)
        #
        #
        # novel_means = []
        # novel_minus_means = []
        # novel_path = '/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/novel_features.plk'
        # with open(novel_path, 'rb') as f:
        #     data = pickle.load(f)
        #     index = 0
        #     for key in data.keys():
        #         feature = np.array(data[key])
        #
        #         mean = np.mean(feature, axis=0)
        #         novel_means.append(mean)
        #
        #         feature = torch.Tensor(feature).cuda()
        #
        #         cos_val, sin_val = get_cos_sin(feature, base_means)
        #         feature = F.normalize((feature - cos_val*base_means)).cpu().numpy()
        #         novel_minus_means.append(np.mean(feature, axis=0))
        #
        #
        # cls_name = args.cls
        # cls_num = k
        #
        # import matplotlib.pyplot as plt
        # novel_means  = torch.Tensor(np.array(novel_means)).cuda()
        # novel_minus_means = torch.Tensor(np.array(novel_minus_means)).cuda()
        # cos_val, sin_val = get_cos_sin_2(novel_means, novel_means)
        # plt.imshow(cos_val.cpu().numpy(), cmap=plt.cm.Blues, vmin=0, vmax=1)
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=15)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('cosine_old.jpg')
        # plt.savefig('cosine_old.pdf')
        # plt.close()
        # print(cos_val.shape)
        # print(cos_val)
        # cos_val, sin_val = get_cos_sin_2(novel_minus_means, novel_minus_means)
        # plt.imshow(cos_val.cpu().numpy(), cmap=plt.cm.Blues, vmin=0, vmax=1)
        # # plt.colorbar()
        # cb = plt.colorbar()
        # cb.ax.tick_params(labelsize=15)
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('cosine_new.jpg')
        # plt.savefig('cosine_new.pdf')
        # print(cos_val.shape)
        # print(cos_val)


        feature = np.concatenate(np.array(features_list)).reshape(-1, dim)
        # print('hahahahah', feature.shape)
        if cls_num == 'all':
            base_means = torch.Tensor(feature).cuda()
            base_means = torch.mean(base_means, dim=0).unsqueeze(0)
            base_means = F.normalize(base_means).cuda()

            # test_all = np.concatenate([support_data, query_data],axis=0)
            # test_all = torch.Tensor(test_all).cuda()
            # test_all = torch.mean(test_all, dim=0).unsqueeze(0)
            # base_means = F.normalize(test_all).cuda()


    # base_means = torch.Tensor(base_means).cuda()
        else:
            base_means = np.concatenate(np.array(base_means)).reshape(-1, dim)
            base_means = torch.Tensor(feature).cuda()
            base_means = F.normalize(base_means)
            support_tensor = torch.Tensor(support_data)
            support_mean = torch.mean(support_tensor, dim=0).unsqueeze(0)

            similar_support_mean = torch.mm(F.normalize(support_tensor).cuda(), F.normalize(support_mean).cuda().transpose(0,1))

            similar = torch.mm(F.normalize(support_mean).cuda(), F.normalize(base_means).cuda().transpose(0,1))
            _, pred = similar[0].topk(k, 0, True, True)
            base_means = F.normalize(torch.mean(base_means[pred,:], dim=0).unsqueeze(0))
            # base_means = torch.mean(base_means[pred,:], dim=0).unsqueeze(0)
            # base_means = F.normalize(support_mean).cuda()


        # X_aug_mean = np.mean(X_aug, axis=1)
        # X_aug_var = np.std(X_aug, axis=1)
        # X_aug = (X_aug - X_aug_mean.reshape(-1, 1))/X_aug_var.reshape(-1, 1)
        #
        # query_data_mean =np.mean(query_data, axis=1)
        # query_data_var =np.std(query_data, axis=1)
        # query_data = (query_data - query_data_mean.reshape(-1, 1))/query_data_var.reshape(-1, 1)

        if args.fc_classifier:

            # linear_clf = get_linear_clf('cosine', 640, n_ways, margin=0.0, scale_factor=5).cuda()
            linear_clf = get_linear_clf(cls_name, dim, n_ways, margin=0.0, scale_factor=5, beta=cls_beta, concat=args.concat).cuda()

            finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), lr=0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
            loss_function = nn.CrossEntropyLoss().cuda()
            batch_size = 4
            support_size = X_aug.shape[0]
            X_aug = torch.Tensor(X_aug).cuda()
            Y_aug = torch.LongTensor(Y_aug).cuda()
            query_data = torch.Tensor(query_data).cuda()
            query_label = torch.LongTensor(query_label).cuda()

            # from models.proo_head import L2SquareDist
            # dis = L2SquareDist(X_aug.unsqueeze(0), base_means.unsqueeze(0), base_means=None, average=False)


            for _ in range(100):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size, batch_size):

                    selected_id = torch.from_numpy(rand_id[i: min(i + batch_size, support_size)]).cuda()
                    z_batch = X_aug[selected_id]
                    y_batch = Y_aug[selected_id]
                    # scores = linear_clf(z_batch)
                    if cls_name == 'new':
                        scores, _ = linear_clf(z_batch, base_means)
                    else:
                        scores = linear_clf(z_batch)

                    loss = loss_function(scores, y_batch)

                    finetune_optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    finetune_optimizer.step()

            if cls_name == 'new':
                scores, _ = linear_clf(query_data, base_means)
            else:
                scores = linear_clf(query_data)
            acc = accuracy(scores, query_label).detach().cpu().numpy()


            # dis = dis.cpu().numpy()
            # dis = dis.mean()
            # dis = np.around(dis, 1)
            # if dis in acc_dis.keys():
            #     acc_dis[dis].append(acc)
            # else:
            #     acc_dis[dis] = [acc]

        # ---- train classifier
        else:
            # X_aug_1 = torch.Tensor(X_aug).cuda()
            #
            X_aug = torch.Tensor(X_aug).cuda()
            query_data = torch.Tensor(query_data).cuda()

            cos_val, sin_val = get_cos_sin(X_aug, base_means)
            X_aug = F.normalize((X_aug - cos_val*base_means)).cpu().numpy()
            # X_aug_1 = F.normalize(X_aug_1).cpu().numpy()
            #
            cos_val, sin_val = get_cos_sin(query_data, base_means)
            query_data = F.normalize((query_data - cos_val*base_means)).cpu().numpy()
            #
            # #
            # X_aug = F.normalize(X_aug).cpu().numpy()
            # query_data = F.normalize((query_data)).cpu().numpy()
            #
            # X_aug = np.concatenate([X_aug, X_aug_1])
            # Y_aug = np.concatenate([support_label, support_label])


            if cls_name == 'PN':
                query_data = torch.Tensor(query_data).unsqueeze(0).cuda()
                support_data = torch.Tensor(X_aug).unsqueeze(0).cuda()
                classifier = PN_head().cuda()

                classification_scores = classifier(query_data, support_data,  n_ways, n_shot)   # shape (batch, num_, n_way)
                cls = torch.argmax(classification_scores.squeeze(0), dim=1)
                acc = np.mean(cls.detach().cpu().numpy() == query_label)



            if cls_name == 'LR':
                classifier = LogisticRegression(max_iter=1000, solver='liblinear', n_jobs=1).fit(X=X_aug, y=Y_aug)
                predicts = classifier.predict(query_data)
                acc = np.mean(predicts == query_label)


            # classifier = LogisticRegression(max_iter=1000, solver='liblinear', n_jobs=1).fit(X=X_aug, y=Y_aug)
            # predicts = classifier.predict(query_data)
            # acc = np.mean(predicts == query_label)

        acc_list.append(acc)
    # for key in acc_dis:
    #     print('dis is:', dis, 'acc is:', np.array(acc_dis[key]).mean())
    # with open('/data/jingxu/project/fsl_distribution/pickle_file/acc_dis_cosine_28', 'wb') as f:
    #     pickle.dump(acc_dis, f)

    print(mean_confidence_interval(acc_list))
    writer.add_scalar('acc', np.mean(acc_list), step)
    writer.add_scalar('def_sampled', np.mean(dif_sampled_list), step)
    writer.add_scalar('def_original_1', np.mean(dif_original_list_1), step)
    writer.add_scalar('def_original_2', np.mean(dif_original_list_2), step)


    # print('miniImageNet %d way %d shot  ACC : %f best ACC is: %f' % (n_ways, n_shot, float(np.mean(acc_list)), BEST_ACC))
    print('miniImageNet %d way %d shot  ACC : %f best ACC is: %f, dif_sampled is: %f, dis_original_1 is: %f,  dis_original_2 is: %f,' % (n_ways, n_shot, float(np.mean(acc_list)), BEST_ACC, float(np.mean(dif_sampled_list)), float(np.mean(dif_original_list_1)), float(np.mean(dif_original_list_2))))
    print('dataset is:', dataset_name, 'k is:', k, 'cls_type is:', cls_name, 'cls_hyper is:', cls_beta)


def main_train(args, select_class=None):

    dataset_name = 'miniImagenet'    #'miniImagenet or tiered_imagenet'
    n_shot = 10
    n_ways = 10
    n_queries = 15
    num_sampled = 1
    # n_rus = 10000
    n_runs = 2000

    n_lsamples = n_ways * n_shot
    n_usamples = n_ways * n_queries
    n_samples = n_lsamples + n_usamples
    train_epoch = 1
    suffix = args.suffix


    tsne = TSNE()

    import FSLTask
    cfg = {'shot': n_shot, 'ways': n_ways, 'queries': n_queries}

    FSLTask.loadDataSet('miniImagenet')
    # FSLTask.loadDataSet('baseline++zscore', is_hdf5=True)


    FSLTask.setRandomStates(cfg)
    # ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg, select_class=[2,3,4,5,6])
    ndatas = FSLTask.GenerateRunSet(end=n_runs, cfg=cfg, select_class=select_class)
    ndatas = ndatas.permute(0, 2, 1, 3).reshape(n_runs, n_samples, -1)  # shape [10000, 80, 640]
    labels = torch.arange(n_ways).view(1, 1, n_ways).expand(n_runs, n_shot + n_queries, n_ways).clone().view(n_runs,n_samples)

    # ---- Base class statistics
    base_means = []
    base_cov = []
    base_keys = []
    # model_name = 'ResNet18'   # Conv4 1600, ResNet12 640, ResNet18 512
    # base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/%s/%s_baseline++_aug/base.hdf5" % (dataset, model_name)

    base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/%s/WideResNet28_10_S2M2_R/base_features.plk" % dataset_name
    # base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.9_aug/base.hdf5"
    # base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/CUB/WideResNet28_10_S2M2_R/base_features.plk"
    # base_features_path = "/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_9s/base_features_new.plk"
    if dataset_name == 'miniImagenet':
        dataset = SimpleDataset_VAE(base_features_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True,
                                             num_workers=4, pin_memory=True)

        class_num = 64
        latent_dim = 256
        hidden_dim = 4096
        lr = 0.0001
        

    elif dataset_name == 'CUB':
        dataset = CUBDateset_VAE(base_features_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True,
                                             num_workers=4, pin_memory=True)
        class_num = 100
        latent_dim = 256
        hidden_dim = 2048
        lr = 0.0002

    elif dataset_name == 'tiered_imagenet':

        dataset = CUBDateset_VAE(base_features_path)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True,
                                             num_workers=4, pin_memory=True)
        class_num = 351
        latent_dim = 128
        hidden_dim = 256
        lr = 0.0001





    # 搭配(32, 256)或者(32, 128都可)
    linear_vae = LinearVAE(640, latent_dim=latent_dim, hidden_dim=hidden_dim, class_num=class_num)
    linear_vae = linear_vae.cuda()

    optimizer = torch.optim.Adam(linear_vae.parameters(),
                                 lr=lr) # 0.0001

    linear_vae_base = LinearVAE(640, latent_dim=latent_dim, hidden_dim=hidden_dim, class_num=class_num)
    linear_vae_base = linear_vae_base.cuda()
    linear_vae_base.eval()


    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma= 0.95)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.5)

    writer = SummaryWriter('./output/{}_ways_{}_shots_{}_sampled_{}'.format(n_ways, n_shot, num_sampled, suffix))

    for epoch in range(train_epoch):  # train_epoch
        # --- 训练vae
        beta = 1
        # for k in ['all', 10, 50, 100, 500, 1000, 5000, 10000]:
        for k in ['all',10000]:
            for beta in [1]:
                evaluate(args, linear_vae, linear_vae_base, n_runs, ndatas, labels, n_lsamples=n_lsamples, n_usamples=n_usamples, num_sampled=1, n_ways=n_ways, n_shot=n_shot, writer=writer, step=len(loader)*epoch, dataset_name=dataset_name, k=k, cls_beta=beta)
        # train(linear_vae, linear_vae_base, loader, optimizer, writer, epoch, scheduler, ndatas, labels, base_means, n_lsamples, n_usamples, n_runs, dataset)

        # scheduler.step()




if __name__ == '__main__':
    # ---- data loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_classifier', action='store_true', default=False,
                        help='using fc classifyer')
    parser.add_argument('--suffix' , type=str, default='None', help='save file suffix')
    parser.add_argument('--cls' , type=str, default='cosine', help='save file suffix')
    parser.add_argument('--concat', action='store_true', default=False,
                        help='using fc classifyer')


    args = parser.parse_args()
    # for i in range(19):
    #     i = i+1
    #     select_class = [2]
    #     if i == 2:
    #         continue
    #     select_class.append(i)
    #     print('selected_class is: ', select_class)
    select_class = [8,2]
    select_class = None
    main_train(args, select_class)


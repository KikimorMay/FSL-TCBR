import numpy as np
import matplotlib.pyplot as plt
import torch
from models.proo_head import PN_head
import pickle


from scipy.stats import t
import scipy

def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def get_cos_sin( x, y):
    cos_val = (x * y).sum(-1, keepdim=True) / torch.norm(x, 2, 1, keepdim=True) / torch.norm(y, 2, 1, keepdim=True)
    sin_val = (1 - cos_val * cos_val).sqrt()
    return cos_val, sin_val

def gen_clusters(a, sample_num):
    # mean1 = [1+a, 1, 1, 1]
    # cov1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]]
    # data1 = np.random.multivariate_normal(mean1, cov1, sample_num)
    #
    # mean2 = [1-a, 1, 1, 1]
    # cov2 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0,0,0,1]]
    mean1 = [1+a, 1, 1]
    cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, sample_num)

    mean2 = [1-a, 1, 1]
    cov2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    data2 = np.random.multivariate_normal(mean2, cov2, sample_num)

    return np.round(data1, 4), np.round(data2, 4)


def get_acc(a, n_shot, n_query=200, use_tcpr=True, metric='cosine'):
    acc_dict = {}
    acc_list = []
    dis_dict = {}
    n_task = 10000
    for i in range(n_task):
        class1, class2 = gen_clusters(a, n_shot + n_query)
        class1 = torch.Tensor(class1).unsqueeze(0).cuda()
        class2 = torch.Tensor(class2).unsqueeze(0).cuda()
        label1 = np.array([0] * (n_query))
        label2 = np.array([1] * (n_query))

        if use_tcpr == True:
            base_means = torch.mean(class1+class2, dim=1)
            cos_val, sin_val = get_cos_sin(class1, base_means)
            class1 = class1 - cos_val*base_means

            cos_val, sin_val = get_cos_sin(class2, base_means)
            class2 =  class2 - cos_val*base_means



        class1_s = class1[:, :n_shot, :].reshape(1, n_shot, -1)  # (batch, n_shot, n_dim)
        class1_q = class1[:, n_shot:, :]
        class2_s = class2[:, :n_shot, :].reshape(1, n_shot, -1)
        class2_q = class2[:, n_shot:, :]
        query_label = np.concatenate([label1, label2])

        prototypes_1 = torch.mean(class1_s, dim=1, keepdim=True)
        prototypes_2 = torch.mean(class2_s, dim=1, keepdim=True)

        prototype = torch.cat([prototypes_1, prototypes_2], dim=1)

        support_data = torch.cat([class1_s, class2_s], axis=1)
        query_data = torch.cat([class1_q, class2_q], axis=1)
        classifier = PN_head(scale_cls=1, normalize=False, metric=metric).cuda()
        classification_scores = classifier(query_data, support_data, 2, n_shot,
                                           prototypes=prototype)  # shape (batch, num_, n_way)
        cls = torch.argmax(classification_scores.squeeze(0), dim=1)

        acc = np.mean(cls.detach().cpu().numpy() == query_label)
        acc_list.append(acc)

    acc_np = np.array(acc_list)
    acc, var = mean_confidence_interval(acc_np)
    print('average acc is:', acc, ' var is:', var)



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--a', type=float, default='1',
                    help='0.5/1/2/3')
parser.add_argument('--n_shot', type=int, default='1',
                    help='1/3/5/10')
parser.add_argument('--metric', type=str, default='cosine',
                    help='cosine/euclidean')
parser.add_argument('--tcpr', type=bool, default=True,
                    help='True/False')
args = parser.parse_args()


get_acc(a=args.a, n_shot=args.n_shot, use_tcpr=args.tcpr, metric=args.metric)





import numpy as np
import matplotlib.pyplot as plt
import torch
from models.proo_head import PN_head
import pickle


def gen_clusters(a, sample_num):
    mean1 = [a, 0]
    cov1 = [[1, 0], [0, 1]]
    data1 = np.random.multivariate_normal(mean1, cov1, sample_num)

    mean2 = [-a, 0]
    cov2 = [[1, 0], [0, 1]]
    data2 = np.random.multivariate_normal(mean2, cov2, sample_num)

    return np.round(data1, 4), np.round(data2, 4)


def draw_gaussian_acc(acc_dict, dis_dict, acc_all_mean, a, shot):
    info = acc_dict
    key_sort = sorted(info.keys())
    dis_list = []
    acc_list = []
    acc_std_list = []
    acc_all_list = []
    dis_num = []
    for key in key_sort:
        dis_list.append(key)
        acc_all_list.append(info[key])
        acc_std_list.append(np.std(np.array(info[key])))
        acc_list.append(np.array(info[key]).mean())
        dis_num.append(dis_dict[key])
        print('dis is:', key, 'acc is:', np.array(info[key]).mean())

    dis_list = np.array(dis_list)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.errorbar(dis_list, acc_list, acc_std_list, fmt='o-', ecolor='lightcoral', color='royalblue',
                elinewidth=1.5, ms=3.5, capsize=2, label='Accuracies with Varied Distance')
    ax.plot([dis_list.min(), dis_list.max()], [acc_all_mean, acc_all_mean], color='orange', label='Average Accuracy %s' % np.round(acc_all_mean*100, 1))
    plt.legend(fontsize=12, loc='center right')

    ax.set_ylim((0.0, 1.05))
    ax.set_xlim((dis_list.min(), dis_list.max()))

    ax.set_yticks(np.arange(0, 11) / 10.0, np.arange(0, 11) * 10, fontsize=10, color='indianred')
    ax.set_ylabel("Accuracy(%)", fontsize=15, color='indianred')
    ax.set_xlabel("Average Distance with Local Centorid", fontsize=15)

    ax1 = ax.twinx()
    ax1.set_ylim((0.0, 1.05))
    ax1.bar(dis_list, np.array(dis_num) / 1000, alpha=0.3, width=0.1, color='g', label='# Number of Tasks')
    ax1.set_yticks(np.arange(0, 11) / 10.0, np.arange(0, 11) * 100, fontsize=10, color='g')
    ax1.set_ylabel("#Numbers of Tasks", fontsize=15, color='g')

    plt.legend(fontsize=15, loc='lower right')
    plt.show()


def get_acc(a, n_shot, n_query=200):
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

        class1_s = class1[:, :n_shot, :].reshape(1, n_shot, -1)  # (batch, n_shot, n_dim)
        class1_q = class1[:, n_shot:, :]
        class2_s = class2[:, :n_shot, :].reshape(1, n_shot, -1)
        class2_q = class2[:, n_shot:, :]
        query_label = np.concatenate([label1, label2])

        prototypes_1 = torch.mean(class1_s, dim=1, keepdim=True)
        prototypes_2 = torch.mean(class2_s, dim=1, keepdim=True)

        prototype = torch.cat([prototypes_1, prototypes_2], dim=1)

        dis = (torch.norm(prototypes_1, p=2) + torch.norm(prototypes_2, p=2))

        support_data = torch.cat([class1_s, class2_s], axis=1)
        query_data = torch.cat([class1_q, class2_q], axis=1)
        classifier = PN_head(scale_cls=1, normalize=False, metric="euclidean").cuda()
        classification_scores = classifier(query_data, support_data, 2, n_shot,
                                           prototypes=prototype)  # shape (batch, num_, n_way)
        cls = torch.argmax(classification_scores.squeeze(0), dim=1)

        acc = np.mean(cls.detach().cpu().numpy() == query_label)
        acc_list.append(acc)

        if i % 200 == 0:
            print('step is :', i)

        dis = dis.cpu().numpy()
        dis = dis.mean()
        dis = np.around(dis, 1)
        if dis in dis_dict:
            dis_dict[dis] = dis_dict[dis] + 1
        else:
            dis_dict[dis] = 1

        if dis in acc_dict.keys():
            acc_dict[dis].append(acc)
        else:
            acc_dict[dis] = [acc]

    acc_np = np.array(acc_list)
    draw_gaussian_acc(acc_dict, dis_dict, acc_np.mean(), a, n_shot)


get_acc(a=1.5, n_shot=1)



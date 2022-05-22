import numpy as np
import matplotlib.pyplot as plt
import torch
from models.proo_head import PN_head
import pickle

def gen_clusters(a, sample_num):
    # mean1 = [1,-1, 1]
    # cov1 = [[0.5,0, 0],[0,1, 0], [0, 0, 0.5]]
    # data1 = np.random.multivariate_normal(mean1,cov1,150)
    #
    #
    # mean2 = [1,1, 1]
    # cov2 =  [[0.5,0, 0],[0,1, 0], [0, 0, 0.5]]
    # data2 = np.random.multivariate_normal(mean2,cov2,150)
    mean1 = [a,0]
    cov1 = [[1,0],[0,1]]
    data1 = np.random.multivariate_normal(mean1,cov1,sample_num)


    mean2 = [-a,0]
    cov2 =  [[1,0],[0,1]]
    data2 = np.random.multivariate_normal(mean2,cov2,sample_num)

    return np.round(data1,4), np.round(data2,4)


def draw_gaussian_acc(acc_dict, dis_dict, acc_all_mean):

    import pickle
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



    print(np.around(np.array(acc_list).mean(),4))
    fig,ax = plt.subplots(figsize=(7,3.6))
    ax.errorbar(dis_list[1:-4], acc_list[1:-4], acc_std_list[1:-4], fmt='o-', ecolor='lightcoral', color='royalblue', elinewidth=1.5, ms=3.5,capsize=2, label='Accuracies with Varied Distance')
    ax.plot([0.2,7.5], [acc_all_mean,acc_all_mean], color='orange', label='Average Accuracy(60.62%)')
    # ax.legend(fontsize=15)
    plt.legend(fontsize=12, loc='upper right')

    # print()

    # plt.plot(dis_list[::2], acc_list[:-3])
    # plt.plot(dis_list[:-3], acc_list[:-3])

    # plt.ylim((0, 2000))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlim((0.2,7.5))
    # plt.legend(fontsize=15,loc = 'upper left')
    # plt.xticks([0.7, 0.8, 0.9, 1.0],fontsize=15)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],fontsize=10,color='indianred')
    ax.set_ylabel("Accuracy(%)", fontsize=15,color='indianred')
    ax.set_xlabel("Average Distance with Local Centorid", fontsize=15)

    ax1=ax.twinx() # 设置双坐标轴
    ax1.set_ylim((0.0, 1.05))
    ax1.bar(dis_list[1:-4], np.array(dis_num)[1:-4]/1000, alpha=0.3, width=0.1, color='g',  label='# Number of Tasks')
    ax1.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],fontsize=10, color='g')
    ax1.set_ylabel("#Numbers of Tasks", fontsize=15, color='g')
    # ax1.legend(fontsize=15)


    plt.legend(fontsize=15, loc='lower right')
    plt.savefig('tsne/dis_gaussian_a05s1.jpg')
    plt.savefig('tsne/dis_gaussian_a05s1.pdf')



def get_acc(a, n_shot, n_query=200):
    acc_dict = {}
    acc_list = []
    dis_dict = {}
    for i in range(10000):
        class1,  class2 = gen_clusters(a, n_shot + n_query)
        class1 = torch.Tensor(class1).unsqueeze(0).cuda()
        class2 = torch.Tensor(class2).unsqueeze(0).cuda()
        label1 = np.array([0]*(n_query))
        label2 = np.array([1]*(n_query))

        class1_s = class1[:, :n_shot, :].reshape(1, n_shot, -1) # (batch, n_shot, n_dim)
        class1_q = class1[:, n_shot:, :]
        class2_s = class2[:, :n_shot, :].reshape(1, n_shot, -1)
        class2_q = class2[:, n_shot:, :]
        query_label = np.concatenate([label1, label2])


        prototypes_1 = torch.mean(class1_s, dim=1, keepdim=True)
        prototypes_2 = torch.mean(class2_s, dim=1, keepdim=True)

        prototype = torch.cat([prototypes_1, prototypes_2], dim=1)

        dis =  (torch.norm(prototypes_1, p=2) + torch.norm(prototypes_2, p=2))

        support_data = torch.cat([class1_s, class2_s], axis=1)
        query_data = torch.cat([class1_q, class2_q], axis=1)
        classifier = PN_head(scale_cls=1, normalize=False, metric="euclidean").cuda()
        classification_scores = classifier(query_data, support_data, 2,  n_shot, prototypes=prototype)   # shape (batch, num_, n_way)
        cls = torch.argmax(classification_scores.squeeze(0), dim=1)

        acc = np.mean(cls.detach().cpu().numpy() == query_label)
        acc_list.append(acc)
        # print(acc)
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
    print(acc_np.mean())
    print(acc_np.std())
    for key in dis_dict:
        print(dis_dict)

    draw_gaussian_acc(acc_dict, dis_dict, acc_np.mean())









get_acc(a=0.2, n_shot=1)

data1, data2 = gen_clusters(1, 100)
# save_data(data,'3clusters.txt')
# d = load_data('3clusters.txt')
print('haha')
# show_scatter(data1, data2)


import pickle
import numpy as np
import matplotlib.pyplot as plt

file_name = 'acc_dis_new_[2, 3]'   # 'acc_dis_cosine_[2,3]'

with open('./pickle_file/' + file_name, 'rb') as f:
    info = pickle.load(f)
key_sort = sorted(info.keys())
dis_list = []
acc_list = []
acc_std_list = []
acc_all_list = []
for key in key_sort:
    dis_list.append(key)
    acc_all_list.append(info[key])
    acc_std_list.append(np.std(np.array(info[key])))
    acc_list.append(np.array(info[key]).mean())
    print('dis is:', key, 'acc is:', np.array(info[key]).mean())
acc_mean = np.around(np.array(acc_list).mean(),4)
print(acc_mean)
plt.figure(figsize=(7,5))
plt.errorbar(dis_list[::2], acc_list[::2], acc_std_list[::2], fmt='o-', ecolor='lightcoral', color='royalblue', elinewidth=2.5, capsize=3, label='Accuracies with Varied Distance')
plt.plot([2,11], [acc_mean,acc_mean], color='orange', label='Average Accuracy '+str(np.round(acc_mean*100,2)) + '%')
plt.legend(fontsize=15)


plt.ylim((0.5, 1.02))
plt.yticks([0.6, 0.7, 0.8, 0.9, 1], [60, 70, 80, 90, 100],fontsize=15)
plt.ylabel("Accuracy(%)", fontsize=15)
plt.xlabel("Average Distance with Local Centorid", fontsize=15)

plt.savefig('./pickle_file/' + file_name + '.jpg')
plt.savefig('./pickle_file/' + file_name + '.pdf')





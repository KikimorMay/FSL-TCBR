import os
import pickle
import numpy as np
import torch
import h5py

# from tqdm import tqdm


# ========================================================
#   Usefull paths
_datasetFeaturesFiles = {"miniImagenetnovel": "/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/novel_features.plk",
                         "miniImagenetbase": "/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R/base_features.plk",
                         'mini2leafnovel': '/data/jingxu/project/fsl_distribution/checkpoints/mini2leaf/last/novel_features.plk',
                         'mini2CUBnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2CUB/last/novel_features.plk',
                         'mini2dtdnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2dtd/last/novel_features.plk',
                         'mini2flowersnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2flowers/last/novel_features.plk',
                         'mini2funginovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2fungi/last/novel_features.plk',
                         'mini2ominovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2omi/last/novel_features.plk',
                         'res12distillnovel':'/data/jingxu/project/fsl_inv_equ/save_features/ResNet12_in_eq_distill/novel_features.plk',
                         'res12nodistillnovel':'/data/jingxu/project/fsl_distribution/checkpoints/eq_in/no_distill/novel_features.plk',
                         'mini2coconovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2coco/s2m2_coco_new.plk',
                         'mini2qdnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2qd/s2m2_QuickD_new.plk',
                         'miniin_eqnovel':'/data/jingxu/project/fsl_inv_equ/save_features/ResNet12_in_eq_distill/novel_features.plk',
                         'mini2airnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2air/last/novel_features.plk',
                         'mini2quickdnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2quickd/last/novel_features.plk',
                         'mini2gtsrbnovel':'/data/jingxu/project/fsl_distribution/checkpoints/mini2gtsrb/last/novel_features.plk',
                         "CUBnovel": "./checkpoints/CUB/WideResNet28_10_S2M2_R/novel_features.plk",
                         'far100novel':"/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/100/far_features.plk",
                         'near100novel':"/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/100/near_features.plk",
                         'far300novel':"/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/300/far_features.plk",
                         'near300novel':"/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/300/near_features.plk",
                         "tiered_imagenetnovel":"/data/jingxu/project/fsl_distribution/checkpoints/tiered_imagenet/WideResNet28_10_S2M2_R/last/novel_features.plk",
                         'baseline++novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_aug/novel.hdf5',
                         'baseline++0.9novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.9_aug/novel.hdf5',
                         'baseline++0.8novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.8_aug/novel.hdf5',
                         'baseline++0.85novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.85_aug/novel.hdf5',
                         'baseline++0.95novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.95_aug/novel.hdf5',
                         'baseline++0.0novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.0_aug/novel.hdf5',
                         'baseline++0.1novel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.1_aug/novel.hdf5',
                         'baseline++zscorenovel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/ResNet18_baseline++_0.9_zscore_aug/novel.hdf5',
                         'Pico01novel':"/data/jingxu/project/PiCO/experiment/PiCO-CIFAR-10/ds_miniImagenet_pr_0.1_lr_0.02_ep_1000_ps_1_lw_0.5_pm_0.99_arch_resnet12_heir_False_sd_None/900_features.plk",
                         'Pico0novel':"/data/jingxu/project/PiCO/experiment/PiCO-CIFAR-10/ds_miniImagenet_pr_0.0_lr_0.02_ep_1000_ps_1_lw_0.5_pm_0.99_arch_resnet12_heir_False_sd_None/900_features.plk",
                         'Pico02novel':"/data/jingxu/project/PiCO/experiment/PiCO-CIFAR-10/ds_miniImagenet_pr_0.2_lr_0.02_ep_500_ps_1_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_None/400_features.plk",
                         'Pico005novel':"/data/jingxu/project/PiCO/experiment/PiCO-CIFAR-10/ds_miniImagenet_pr_0.05_lr_0.02_ep_800_ps_1_lw_0.5_pm_0.99_arch_resnet18_heir_False_sd_None/500_features.plk",
                         'miniImagenetminusnovel':'/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/novel_miuns_centers.pickle',
                         'miniImagenetminus0.7novel':'/data/jingxu/project/fsl_distribution/checkpoints/cos_distance/novel_miuns_centers_0.7.pickle',
                         'miniminus_train16novel':'/data/jingxu/project/fsl_baseline/features/miniImagenet/ResNet18_baseline++_0.9_16_aug/novel.hdf5',
                         'miniminus_train128novel':'/data/jingxu/project/fsl_baseline/features/miniImagenet/ResNet18_baseline++_0.9_128_aug/novel.hdf5',
                         'miniImagenet9snovel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_9s/novel_features_new.plk',
                         'miniImagenetscorenovel':'/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_zscore/novel_features.plk',
                         }
_cacheDir = "/data/jingxu/project/fsl_distribution/cache"
_maxRuns = 10000
_min_examples = -1


# ========================================================
#   Module internal functions and variables

_randStates = None
_rsCfg = None

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

def _load_pickle(file, is_hdf5=False):
    if is_hdf5:
        print(file)
        with h5py.File(file, 'r') as f:
            fileset = SimpleHDF5Dataset(f)
        feats = fileset.all_feats_dset
        labels = fileset.all_labels
        while np.sum(feats[-1]) == 0:
            feats  = np.delete(feats,-1,axis = 0)
            labels = np.delete(labels,-1,axis = 0)
        dataset = dict()
        dataset['data'] = torch.FloatTensor(feats)
        dataset['labels'] = torch.LongTensor(labels)
        return dataset
    else:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            labels = [np.full(shape=len(data[key]), fill_value=key)
                      for key in data]
            data = [features for key in data for features in data[key]]
            dataset = dict()
            dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))  # [12000, 640]
            dataset['labels'] = torch.LongTensor(np.concatenate(labels))  #[12000]
            # dataset = dict()
            # dataset['data'] = torch.FloatTensor(data[0])
            # dataset['labels'] = torch.FloatTensor(data[1])
            return dataset

#
#
# def _load_pickle(file):
#     with open(file, 'rb') as f:
#         data = pickle.load(f)
#         labels = [np.full(shape=len(data[key]), fill_value=key)
#                   for key in data]
#         data = [features for key in data for features in data[key]]
#         dataset = dict()
#         dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))
#         dataset['labels'] = torch.LongTensor(np.concatenate(labels))
#         # dataset['data'] = torch.FloatTensor(data[0])
#         # dataset['labels'] = torch.FloatTensor(data[1])
#
#         return dataset


# =========================================================
#    Callable variables and functions from outside the module

data = None
labels = None
dsName = None


def loadDataSet(dsname, type = 'novel', is_hdf5=False):
    if dsname+type not in _datasetFeaturesFiles:
        raise NameError('Unknwown dataset: {}'.format(dsname))

    global dsName, data, labels, _randStates, _rsCfg, _min_examples
    dsName = dsname
    _randStates = None
    _rsCfg = None

    # Loading data from files on computer
    # home = expanduser("~")
    print(dsname+type)
    dataset = _load_pickle(_datasetFeaturesFiles[dsname+type], is_hdf5)

    # Computing the number of items per class in the dataset
    _min_examples = dataset["labels"].shape[0]
    for i in range(dataset["labels"].shape[0]):
        if torch.where(dataset["labels"] == dataset["labels"][i])[0].shape[0] > 0:
            _min_examples = min(_min_examples, torch.where(
                dataset["labels"] == dataset["labels"][i])[0].shape[0])
    print("Guaranteed number of items per class: {:d}\n".format(_min_examples))

    # Generating data tensors
    data = torch.zeros((0, _min_examples, dataset["data"].shape[1]))  # shape(0, 600, 640)
    labels = dataset["labels"].clone()
    while labels.shape[0] > 0:
        indices = torch.where(dataset["labels"] == labels[0])[0]
        data = torch.cat([data, dataset["data"][indices, :]
                          [:_min_examples].view(1, _min_examples, -1)], dim=0)
        indices = torch.where(labels != labels[0])[0]
        labels = labels[indices]
    print("Total of {:d} classes, {:d} elements each, with dimension {:d}\n".format(
        data.shape[0], data.shape[1], data.shape[2]))    # data shape (20, 600, 640)


def GenerateRun(iRun, cfg, regenRState=False, generate=True, select_class=None):
    global _randStates, data, _min_examples
    if not regenRState:
        np.random.set_state(_randStates[iRun])
    if select_class !=None :
        classes = np.random.permutation(np.array(select_class))[:cfg["ways"]]
    else:
        classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    shuffle_indices = np.arange(_min_examples)  # 600
    dataset = None
    if generate:
        dataset = torch.zeros(
            (cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for i in range(cfg['ways']):
        shuffle_indices = np.random.permutation(shuffle_indices)
        if generate:
            dataset[i] = data[classes[i], shuffle_indices,
                              :][:cfg['shot']+cfg['queries']]

    return dataset


def ClassesInRun(iRun, cfg):
    global _randStates, data
    np.random.set_state(_randStates[iRun])

    classes = np.random.permutation(np.arange(data.shape[0]))[:cfg["ways"]]
    return classes


def setRandomStates(cfg):
    global _randStates, _maxRuns, _rsCfg
    if _rsCfg == cfg:
        return

    rsFile = os.path.join(_cacheDir, "RandStates_{}_s{}_q{}_w{}".format(
        dsName, cfg['shot'], cfg['queries'], cfg['ways']))
    if not os.path.exists(rsFile):
        print("{} does not exist, regenerating it...".format(rsFile))
        np.random.seed(0)
        _randStates = []
        for iRun in range(_maxRuns):
            _randStates.append(np.random.get_state())
            GenerateRun(iRun, cfg, regenRState=True, generate=False)
        torch.save(_randStates, rsFile)
    else:
        print("reloading random states from file....")
        _randStates = torch.load(rsFile)
    _rsCfg = cfg


def GenerateRunSet(start=None, end=None, cfg=None, select_class=None):
    global dataset, _maxRuns
    if start is None:
        start = 0
    if end is None:
        end = _maxRuns
    if cfg is None:
        cfg = {"shot": 1, "ways": 5, "queries": 15}

    setRandomStates(cfg)
    print("generating task from {} to {}".format(start, end))

    dataset = torch.zeros(
        (end-start, cfg['ways'], cfg['shot']+cfg['queries'], data.shape[2]))
    for iRun in range(end-start):
        dataset[iRun] = GenerateRun(start+iRun, cfg, select_class=select_class)

    return dataset



def change_format(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        print(data[0].shape)
        print(data[1])
        num = data[1].shape
        import collections

        output_dict = collections.defaultdict(list)


        for out, label in zip(data[0], data[1]):
            output_dict[label.item()].append(out)

        print(output_dict.keys())
        path = '/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_9s/novel_features_new.plk'
        with open(path, 'wb') as f:
            pickle.dump(output_dict, f)


        labels = [np.full(shape=len(data[key]), fill_value=key)
                  for key in data]
        data = [features for key in data for features in data[key]]
        dataset = dict()
        dataset['data'] = torch.FloatTensor(np.stack(data, axis=0))  # [12000, 640]
        dataset['labels'] = torch.LongTensor(np.concatenate(labels))  #[12000]
        # dataset = dict()
        # dataset['data'] = torch.FloatTensor(data[0])
        # dataset['labels'] = torch.FloatTensor(data[1])

# define a main code to test this module
if __name__ == "__main__":

    print("Testing Task loader for Few Shot Learning")
    # loadDataSet('miniimagenet', type='novel')


    data_path = '/data/jingxu/project/fsl_distribution/checkpoints/mini2coco/s2m2_coco.plk'
    # data_path = '/data/jingxu/project/fsl_distribution/checkpoints/mini2qd/s2m2_QuickD.plk'

    # data_path = '/data/jingxu/project/fsl_distribution/checkpoints/mini2air/last/novel_features.plk'
    # data_path = '/data/jingxu/project/fsl_distribution/checkpoints/mini2quickd/last/novel_features.plk'
    data_path = '/data/jingxu/project/fsl_distribution/checkpoints/miniImagenet/WideResNet28_10_S2M2_R_9s/novel_features.plk'

    change_format(data_path)

    loadDataSet('baseline++')


    cfg = {"shot": 1, "ways": 5, "queries": 15}
    setRandomStates(cfg)

    run10 = GenerateRun(10, cfg)
    print("First call:", run10[:2, :2, :2])

    run10 = GenerateRun(10, cfg)
    print("Second call:", run10[:2, :2, :2])

    ds = GenerateRunSet(start=2, end=12, cfg=cfg)
    print("Third call:", ds[8, :2, :2, :2])
    print(ds.size())

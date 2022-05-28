# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import pickle
import numpy as np
import torchvision.transforms as transforms
import os
identity = lambda x:x
from torch.utils.data import Dataset
import os.path as osp



class tieredImageNet(Dataset):
    def __init__(self, setname, aug=False, args=None):
        TRAIN_PATH = osp.join('./dataset', 'tiered_imagenet/train')
        VAL_PATH = osp.join('./dataset', 'tiered_imagenet/val')
        TEST_PATH = osp.join('./dataset', 'tiered_imagenet/test')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        elif setname == 'test':
            THE_PATH = TEST_PATH
        elif setname == 'val':
            THE_PATH = VAL_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class leafDisease(Dataset):
    def __init__(self, setname, aug=False, args='leaf_disease/train'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        if args == 'fungi/images':
            for idx in range(len(folders)):
                this_folder = folders[idx]
                this_folder_images = os.listdir(this_folder)
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(idx)
        else:
            lab = 0
            for idx in range(len(folders)):
                this_folder = folders[idx]
                this_folder_images = os.listdir(this_folder)
                if len(this_folder_images) > 25:
                    for image_path in this_folder_images:
                        data.append(osp.join(this_folder, image_path))
                        label.append(lab)
                    lab = lab + 1

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class Omniglot(Dataset):
    def __init__(self, setname, aug=False, args='images_background'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []

        folders = []
        for suffix1 in os.listdir(THE_PATH):
            path2 = osp.join(THE_PATH, suffix1)
            for suffix2 in os.listdir(path2):
                path3 = osp.join(path2, suffix2)
                folders.append(path3)


        lab = 0
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            print(this_folder_images)
            if len(this_folder_images) > 19:
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(lab)
                lab = lab + 1
        print(len(data))
        print(len(label))

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class Omniglot(Dataset):
    def __init__(self, setname, aug=False, args='images_background'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []

        folders = []
        for suffix1 in os.listdir(THE_PATH):
            path2 = osp.join(THE_PATH, suffix1)
            for suffix2 in os.listdir(path2):
                path3 = osp.join(path2, suffix2)
                folders.append(path3)


        lab = 0
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            print(this_folder_images)
            if len(this_folder_images) > 19:
                for image_path in this_folder_images:
                    data.append(osp.join(this_folder, image_path))
                    label.append(lab)
                lab = lab + 1

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class quickdraw(Dataset):
    def __init__(self, setname, aug=False, args='quickdraw'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []

        folders = []
        for suffix1 in os.listdir(THE_PATH):
            path2 = osp.join(THE_PATH, suffix1)
            if os.path.isdir(path2):
                folders.append(path2)

        lab = 0
        print(folders)
        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)

            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(lab)
            lab = lab + 1


        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class airplane(Dataset):
    def __init__(self, setname, aug=False, args='fgvc-aircraft-2013b/data'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        txt_file = './dataset/fgvc-aircraft-2013b/data/images_family_train.txt'
        fh = open(txt_file, 'r')

        folders = []
        target = []

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append(TRAIN_PATH + '/images/'+ words[0] + '.jpg')
            target.append(''.join(words[1:]))
            # print(target)
            # print(data)

        lab = 0
        cls_pre = target[0]
        for cls in target:
            if cls_pre != cls:
                lab = lab + 1
                print(cls_pre, '+' , cls)
            cls_pre = cls
            label.append(lab)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class coco(Dataset):
    def __init__(self, setname, aug=False, args='coco'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        json_file = './dataset/coco/annotations/instances_train2017.json'
        fh = open(json_file, 'r')
        json_data=json.load(fh)
        print(json_data)

        folders = []
        target = []

        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            data.append(TRAIN_PATH + '/images/'+ words[0] + '.jpg')
            target.append(''.join(words[1:]))
            # print(target)
            # print(data)

        lab = 0
        cls_pre = target[0]
        for cls in target:
            if cls_pre != cls:
                lab = lab + 1
                print(cls_pre, '+' , cls)
            cls_pre = cls
            label.append(lab)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class flowers(Dataset):
    def __init__(self, setname, aug=False, args='102flowers/jpg'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        from scipy.io import loadmat
        label = loadmat('./dataset/102flowers/imagelabels.mat')["labels"].flatten() - 1

        this_folder_images = os.listdir(TRAIN_PATH)
        this_folder_images.sort()
        for image_path in this_folder_images:
            data.append(osp.join(TRAIN_PATH, image_path))
        print(data[:5])
        print(label[:5])




        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class gtsrb(Dataset):
    def __init__(self, setname, aug=False, args='GTSRB/Final_Training/Images'):
        TRAIN_PATH = osp.join('./dataset', args)
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                (shotname, suffix) = os.path.splitext(image_path)
                if suffix == '.ppm':
                    data.append(osp.join(this_folder, image_path))
                    label.append(idx)
                else:
                    pass

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]

        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class cub(Dataset):
    def __init__(self, setname, aug=False, args=None):
        TRAIN_PATH = osp.join('./dataset', 'CUB_200_2011/images')
        if setname == 'train':
            THE_PATH = TRAIN_PATH
        else:
            raise ValueError('Wrong setname.')
        data = []
        label = []
        folders = [osp.join(THE_PATH, label) for label in os.listdir(THE_PATH) if
                   os.path.isdir(osp.join(THE_PATH, label))]

        for idx in range(len(folders)):
            this_folder = folders[idx]
            this_folder_images = os.listdir(this_folder)
            for image_path in this_folder_images:
                data.append(osp.join(this_folder, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        # Transformation
        if aug == False:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.Resize([92, 92]),
                transforms.CenterCrop(image_size),

                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        elif aug == True:
            image_size = 84
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])

    def __len__(self):
        # return 1000
        return len(self.data)

    def __getitem__(self, i):

        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label



class SimpleDataset_9s:
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            self.meta = pickle.load(f)
        self.data = self.meta[0]
        self.label = self.meta[1]
        print(self.data.shape)
        print(self.label.shape)


    def __getitem__(self,i):
        # keys = i // self.img_per_class
        # num_image = i % self.img_per_class

        # return self.meta[keys][num_image], keys
        return self.data[i,:], self.label[i]


    def __len__(self):
        # return self.img_per_class * len(self.meta.keys())
        # return self.img_per_class * len(self.meta.keys())
        return self.label.shape[0]




class SimpleDataset:
    # def __init__(self, data_file):
    #     with open(data_file, 'rb') as f:
    #         self.meta = pickle.load(f)
    def __init__(self, data_file, hdf5=False):
        if hdf5:
            with h5py.File(data_file, 'r') as f:
                fileset = SimpleHDF5Dataset(f)
            feats = fileset.all_feats_dset
            labels = fileset.all_labels
            while np.sum(feats[-1]) == 0:
                feats  = np.delete(feats,-1,axis = 0)
                labels = np.delete(labels,-1,axis = 0)

            class_list = np.unique(np.array(labels)).tolist()
            inds = range(len(labels))

            self.meta = {}
            for cl in class_list:
                self.meta[cl] = []
            for ind in inds:
                self.meta[labels[ind]].append( feats[ind])
        else:
            with open(data_file, 'rb') as f:
                self.meta = pickle.load(f)

        self.img_per_class = len(self.meta[0])



    def __getitem__(self,i):
        keys = i // self.img_per_class
        num_image = i % self.img_per_class

        return self.meta[keys][num_image], keys

    def __len__(self):
        # return self.img_per_class * len(self.meta.keys())
        return self.img_per_class * len(self.meta.keys())


class CUBDateset:
    def __init__(self, data_file) -> object:
        with open(data_file, 'rb') as f:
            self.meta = pickle.load(f)
        print(self.meta.keys())
        self.data = []
        label = 0
        for key in self.meta.keys():
            for i in range(len(self.meta[key])):
                self.data.append([self.meta[key][i], label])
            label = label + 1

    def __getitem__(self,i):
        return self.data[i][0], self.data[i][1]

    def __len__(self):
        return len(self.data)

class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform
        self.labels = self.meta['image_labels']


    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')
        # img.save('image/original' + str(i) + '.png')
        img = self.transform(img)
        # transforms.ToPILImage()(img).convert('RGB').save('image/'+str(i)+'.png')
        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])


class SetDataset:
    def __init__(self, data_file, batch_size, transform):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
 
        self.cl_list = np.unique(self.meta['image_labels']).tolist()

        self.sub_meta = {}
        for cl in self.cl_list:
            self.sub_meta[cl] = []

        for x,y in zip(self.meta['image_names'],self.meta['image_labels']):
            self.sub_meta[y].append(x)

        self.sub_dataloader = [] 
        sub_data_loader_params = dict(batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 0, #use main thread only or may receive multiple batches
                                  pin_memory = False)        
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform = transform )
            self.sub_dataloader.append( torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params) )

    def __getitem__(self,i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.cl_list)

class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl 
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self,i):
        #print( '%d -%d' %(self.cl,i))
        image_path = os.path.join( self.sub_meta[i])
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]

import h5py


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


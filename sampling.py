import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
from torchvision import datasets, transforms

class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True, BatchorNot = True, frac = 1):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    BatchorNot: if False, the dataloader will give the full length of data instead of a batch, for testing
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        num_data = len(dict_users[i])
        frac_num_data = int(frac*num_data)
        whole_range = range(num_data)
        frac_range = np.random.choice(whole_range, frac_num_data)
        frac_dict_users = [dict_users[i][j] for j in frac_range]
        if BatchorNot== True:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,frac_dict_users),
                        batch_size=len(LocalDataset(dataset,dict_users[i])),
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=True)
        loaders.append(loader)
    return loaders


def partition_data(n_users, alpha=0.5,rand_seed = 0, dataset = 'cifar10'):
    if dataset == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
    if dataset == 'CIFAR100':
        K = 100
        data_dir = '../data/cifar100/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
    if dataset == 'EMNIST':
        K = 62
        data_dir = '../data/EMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.EMNIST(data_dir, train=True, split = 'byclass', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.EMNIST(data_dir, train=False, split = 'byclass', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    if dataset == 'SVHN':
        K = 10
        data_dir = '../data/SVHN/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.SVHN(data_dir, split='train', download=True,
                                       transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split='test', download=True,
                                      transform=apply_transform)
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)
        
    min_size = 0
    N = len(train_dataset)
    N_test = len(test_dataset)
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)
   
    while min_size < 10:
        idx_batch = [[] for _ in range(n_users)]
        idx_batch_test = [[] for _ in range(n_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_users))
            ## Balance
            proportions_train = np.array([p*(len(idx_j)<N/n_users) for p,idx_j in zip(proportions,idx_batch)])
            proportions_test = np.array([p*(len(idx_j)<N_test/n_users) for p,idx_j in zip(proportions,idx_batch_test)])
            proportions_train = proportions_train/proportions_train.sum()
            proportions_test = proportions_test/proportions_test.sum()
            proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
            proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
            idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_users):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        net_dataidx_map_test[j] = idx_batch_test[j]
   
        
#     traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)

    return (train_dataset, test_dataset,net_dataidx_map, net_dataidx_map_test)


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp
    return net_cls_counts

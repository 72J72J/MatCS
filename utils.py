import os
import random
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample
import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CaptionDataset(Dataset):
    def __init__(self, data_folder, data_name,C_data_name, transforms):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        """
        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder,data_name + '.hdf5'), 'r')
        self.imgsH = self.h['images']
        self.c = h5py.File(os.path.join(data_folder, C_data_name + '.hdf5'), 'r')
        self.imgsC = self.c['images']

        # Load encoded captions (completely into memory)
        # Total number of datapoints
        self.transforms = transforms  # transform
        self.cpi = 1


    def __getitem__(self, i):
        img1 = torch.FloatTensor(self.imgsH[i] / 255.)
        if self.transform is not None:
            img1 = self.transform(img1)
        data = (img1, caption, label, img2)
        return data

    def __len__(self):
        return self.dataset_size


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print(1)


def accuracy(output, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    #print('***************************************************************')

    batch_size = targets.size(0)

    ind = torch.zeros_like(output)
    for i in range(output.size(0)):
        if output[i, :] > 0.5:
            ind[i, :] = 1
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    #print(correct)
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total * (100.0 / batch_size)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class TestbedDataset(InMemoryDataset):   #   train_data = TestbedDataset(root='data', dataset=dataset+'_train', xd=train_drugs, xt=train_prots, y=train_Y,smile_graph=smile_graph)
    def __init__(self,data_folder, data_name,C_data_name ,transforms, root='./', dataset='davis',
                 xd=None,  y=None, transform=None,
                 pre_transform=None,smile_graph=None, ):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.h = h5py.File(os.path.join(data_folder, data_name + '.hdf5'), 'r')
        self.imgsH = self.h['images']
        self.c = h5py.File(os.path.join(data_folder, C_data_name + '.hdf5'), 'r')
        self.imgsC = self.c['images']
        self.transforms = transforms

        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            data = self.process(xd, y,smile_graph,self.imgsH,self.imgsC,self.transforms)
            self.data, self.slices = data
            print(self.data)

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, y,smile_graph,imgsH,imgsC,transforms):
        assert (len(xd)  == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        print(data_len)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIII",i+1)
            img1 = torch.FloatTensor(imgsH[i] / 255.)
            img2 = torch.FloatTensor(imgsC[i] / 255.)
            img1 = transforms(img1)
            img2 = transforms(img2)

            smiles = xd[i]
            nmr_H = img1.unsqueeze(0)
            nmr_C = img2.unsqueeze(0)
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            print(c_size)
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))

            GCNData.target = nmr_H
            a = GCNData.x
            b = GCNData.edge_index
            c = GCNData.target

            print("features",a.shape,b.shape,c.shape)
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.__setitem__('nmr_C', nmr_C)
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        data = (data,slices)
        # save preprocessed data:
        return data

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

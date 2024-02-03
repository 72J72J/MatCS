import os
import random
import numpy as np
import h5py
import json
import torch
from torch.utils.data import Dataset


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
    def __init__(self, data_folder, data_name,C_data_name, transform):
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
        with open(os.path.join(data_folder, 'morgenfinger' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, 'label' + data_name + '.json'), 'r') as j:
            self.labels = json.load(j)

        # Total number of datapoints
        self.dataset_size = self.h['images'].shape[0]
        self.transform = transform
        self.cpi = 1

    def __getitem__(self, i):
        img1 = torch.FloatTensor(self.imgsH[i] / 255.)
        img2 = torch.FloatTensor(self.imgsC[i] / 255.)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        caption = torch.LongTensor(self.captions[i])
        label = torch.LongTensor([self.labels[i]])
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


def accuracy(output, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """
    print('***************************************************************')

    batch_size = targets.size(0)
    ind = torch.zeros_like(output)
    for i in range(output.size(0)):
        if output[i, :] > 0.5:
            ind[i, :] = 1
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    print(correct)
    correct_total = correct.view(-1).float().sum()

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
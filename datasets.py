from cgi import test
from doctest import ELLIPSIS_MARKER
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets import split_to_train_test_set
import os
import torch


def get_dataset(dataname, frames_number, data_dir):
    '''
    For a given dataname, return the train and testset

    Parameters:
        dataname (str): name of the dataset

    Returns:
        trainset (torch.utils.data.Dataset): train dataset
        testset (torch.utils.data.Dataset): test dataset
    '''

    '''
    Split_by splits the event to integrate them to frames. This can be done either by setting some fixed time or the number of frames.
    We choose the number of frames following the paper: https://arxiv.org/abs/2007.05785?context=cs.LG

    However the data_type 
    '''

    data_dir = os.path.join(data_dir, dataname)

    if dataname == 'gesture':
        transform = None

        train_set = DVS128Gesture(
            data_dir, train=True, data_type='frame', split_by='number', frames_number=frames_number, transform=transform)
        test_set = DVS128Gesture(data_dir, train=False,
                                 data_type='frame', split_by='number', frames_number=frames_number, transform=transform)

    elif dataname == 'cifar10':

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

        dataset = CIFAR10DVS(data_dir, data_type='frame',
                             split_by='number', frames_number=frames_number)

        cifar = os.join.path(data_dir, 'cifar10.pt')
        if not os.path.exists(cifar):
            # TODO: Since this is slow, consider saving the dataset
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=10)
            torch.save({'train': train_set, 'test': test_set}, cifar)

        else:
            data = torch.load(cifar)
            train_set = data['train']
            test_set = data['test']

    elif dataname == 'mnist':

        train_set = NMNIST(data_dir, train=True, data_type='frame',
                           split_by='number', frames_number=frames_number)

        test_set = NMNIST(data_dir, train=False, data_type='frame',
                          split_by='number', frames_number=frames_number)
    else:
        raise ValueError(f'{dataname} is not supported')

    return train_set, test_set

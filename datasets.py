from cgi import test
from doctest import ELLIPSIS_MARKER
from matplotlib.transforms import Transform
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
    if dataname == 'gesture':
        transform = None

        data_dir = os.path.join(data_dir, 'gesture')
        path = os.path.join(data_dir, 'gesture_dataset.pt')

        if os.path.exists(path):
            dataset = torch.load(path)
            train_set = dataset['train']
            test_set = dataset['test']
        else:

            train_set = DVS128Gesture(
                data_dir, train=True, data_type='frame', split_by='number', frames_number=frames_number, transform=transform)
            test_set = DVS128Gesture(data_dir, train=False,
                                     data_type='frame', split_by='number', frames_number=frames_number, transform=transform)

            torch.save({'train': train_set, 'test': test_set}, path)

    elif dataname == 'cifar10':

        # Split by number as in: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron
        data_dir = os.path.join(data_dir, 'cifar10')

        path = os.path.join(data_dir, 'cifar_dataset.pt')

        if os.path.exists(path):
            dataset = torch.load(path)
            train_set = dataset['train']
            test_set = dataset['test']
        else:
            dataset = CIFAR10DVS(data_dir, data_type='frame',
                                 split_by='number', frames_number=frames_number)

            # TODO: Since this is slow, consider saving the dataset
            train_set, test_set = split_to_train_test_set(
                origin_dataset=dataset, train_ratio=0.9, num_classes=10)

            torch.save({'train': train_set, 'test': test_set}, path)

    elif dataname == 'mnist':

        data_dir = os.path.join(data_dir, 'mnist')

        path = os.path.join(data_dir, 'mnist_dataset.pt')

        if os.path.exists(path):
            dataset = torch.load(path)
            train_set = dataset['train']
            test_set = dataset['test']

        else:

            train_set = NMNIST(data_dir, train=True, data_type='frame',
                               split_by='number', frames_number=frames_number)

            test_set = NMNIST(data_dir, train=False, data_type='frame',
                              split_by='number', frames_number=frames_number)
            torch.save({'train': train_set, 'test': test_set}, path)
    else:
        raise ValueError(f'{dataname} is not supported')

    train_set.samples = train_set.samples[:1000]
    train_set.targets = train_set.targets[:1000]

    test_set.samples = test_set.samples[:200]
    test_set.targets = test_set.targets[:200]
    return train_set, test_set

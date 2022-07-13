from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS


def get_dataset(dataname, frames_number):
    '''
    For a given dataname, return the train and testset

    Parameters:
        dataname (str): name of the dataset

    Returns:
        trainset (torch.utils.data.Dataset): train dataset
        testset (torch.utils.data.Dataset): test dataset
    '''
    if dataname == 'gesture':

        train_set = DVS128Gesture(
            'datasets/gesture', train=True, data_type='frame', split_by='number', frames_number=frames_number)
        test_set = DVS128Gesture('datasets/gesture', train=False,
                                 data_type='frame', split_by='number', frames_number=frames_number)
    elif dataname == 'cifar10':
        train_set = CIFAR10DVS('datasets/cifar10', data_type='frame',
                               split_by='number', frames_number=frames_number)
    else:
        raise ValueError(f'{dataname} is not supported')
    return train_set, test_set

import copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import get_dataset
from PIL import Image
from spikingjelly.datasets import play_frame


class PoisonedDataset(Dataset):
    '''
    Poisoned dataset class

    Attributes:
        dataset (torch.utils.data.Dataset): dataset
        trigger_label (int): label of the target/objective class. The class to be changed to.
        mode (str): 'train' or 'test'
        epsilon (float): rate of poisoned data
        pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
        shape (str): shape of the trigger. 'square' or 'random'
        color (str): color of the trigger. 'white' or 'random'
        trigger_size (float): size of the trigger as the percentage of the image size.
        device (torch.device): device
        dataname (str): name of the dataset
        transform (torchvision.transforms): transform

    Methods:
        __getitem__: returns the poisoned data and the corresponding label
        __len__: returns the length of the dataset
        __shape_info__: returns the shape of the dataset
        reshape: reshapes the dataset
        norm: normalizes the dataset
        add_trigger: adds the trigger to the dataset

    '''

    def __init__(self, dataset, trigger_label=0, mode='train', epsilon=0.1, pos='top-left', trigger_type=0, time_step=16,
                 trigger_size=0.1, device=torch.device('cuda'), dataname='minst'):

        # Handle special case for CIFAR10
        if type(dataset) == torch.utils.data.Subset:
            targets = torch.Tensor(dataset.dataset.targets)[dataset.indices]
            data = np.array([i[0] for i in dataset.dataset])
            data = torch.Tensor(data)[dataset.indices]
            dataset = dataset.dataset
        else:
            targets = dataset.targets
            data = dataset

        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

        self.time_step = time_step
        self.device = device
        self.dataname = dataname
        self.ori_dataset = dataset
        self.transform = dataset.transform

        # TODO: Change the attributes of the imagenet to fit the same as MNIST
        self.data, self.targets = self.add_trigger(
            data, targets, trigger_label, epsilon, mode, pos, trigger_type, trigger_size)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):

        img = self.data[item]
        label_idx = int(self.targets[item])

        if self.transform:
            img = self.transform(img)

        label = np.zeros(self.class_num)
        label[label_idx] = 1  # 把num型的label变成10维列表。
        label = torch.Tensor(label)

        img = img.to(self.device)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[2:]

    def norm(self, data):
        offset = np.mean(data, 0)
        scale = np.std(data, 0).clip(min=1)
        return (data - offset) / scale

    def add_trigger(self, data, targets, trigger_label, epsilon, mode, pos, type, trigger_size):
        '''
        Adds the trigger to the dataset

        Parameters:
            data (torch.tensor): dataset
            targets (torch.tensor): targets
            trigger_label (int): label of the target/objective class. The class to be changed to.
            epsilon (float): rate of poisoned data
            mode (str): 'train' or 'test'
            pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
            trigger_size (float): size of the trigger as the percentage of the image size.

        Returns:
            poisoned_data (torch.tensor): poisoned dataset

        '''

        print("## generate " + mode + " Bad Imgs")
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)

        # Fixes some bugs
        # if not isinstance(torch.Tensor, type(new_targets)):
        if not torch.is_tensor(new_targets):
            new_targets = torch.Tensor(new_targets)

        # Choose a random subset of samples to be poisoned
        perm = np.random.permutation(len(new_data))[
            0: int(len(new_data) * epsilon)]

        frame, label = new_data[0]
        width, height = frame.shape[2:]

        # Swap every samples to the target class
        new_targets[perm] = trigger_label

        size_width = int(trigger_size * width)
        size_height = int(trigger_size * height)

        if pos == 'top-left':
            x_begin = 0
            x_end = size_width
            y_begin = 0
            y_end = size_height

        elif pos == 'top-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = 0
            y_end = size_height

        elif pos == 'bottom-left':

            x_begin = 0
            x_end = size_width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'bottom-right':
            x_begin = int(width - size_width)
            x_end = width
            y_begin = int(height - size_height)
            y_end = height

        elif pos == 'middle':
            x_begin = int((width - size_width) / 2)
            x_end = int((width + size_width) / 2)
            y_begin = int((height - size_height) / 2)
            y_end = int((height + size_height) / 2)

        elif pos == 'random':
            # Note that every sample gets the same (random) trigger position
            # We can easily implement random trigger position for each sample by using the following code
            ''' TODO:
                new_data[perm, :, np.random.randint(
                0, height, size=len(perm)), np.random.randint(0, width, size=(perm))] = value
            '''
            x_begin = np.random.randint(0, width)
            x_end = x_begin + size_width
            y_begin = np.random.randint(0, height)
            y_end = y_begin + size_height

        new_data = np.array([i[0] for i in new_data])

        # Static trigger
        if type == 0:
            # TODO: Take into account the polarity. Being 0 green, 1 ligth blue and 2 a mix of both ie dark blue
            # Check this im not sure
            new_data[perm, :, 0, y_begin:y_end, x_begin:x_end] = 1
            new_data[perm, :, 1, y_begin:y_end, x_begin:x_end] = 0

        else:
            # Dynamic trigger
            new_data = create_dynamic_trigger(
                size_width, size_height, new_data, height, width, perm, pos, self.time_step)
        print(
            f'Injecting Over: Bad Imgs: {len(perm)}. Clean Imgs: {len(new_data)-len(perm)}. Epsilon: {epsilon}')

        return torch.Tensor(new_data), new_targets


def create_backdoor_data_loader(dataname, trigger_label, epsilon, pos, type, trigger_size,
                                batch_size_train, batch_size_test, T, device, data_dir):
    '''
    Creates the data loader for the backdoor training dataset, a clean test dataset, and a fully poisoned test dataset.

    Parameters:
        dataname (str): name of the dataset
        trigger_label (int): label of the target/objective class. The class to be changed to.
        epsilon (float): rate of poisoned data
        pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
        shape (str): shape of the trigger. 'square' or 'random'
        trigger_size (float): size of the trigger as the percentage of the image size.
        batch_size_train (int): batch size for training
        batch_size_test (int): batch size for testing
        device (torch.device): device to use

    Returns:
        train_loader (torch.utils.data.DataLoader): training data loader
        test_loader (torch.utils.data.DataLoader): test data loader
        poisoned_test_loader (torch.utils.data.DataLoader): poisoned test data loader

    '''

    # Get the dataset
    train_data, test_data = get_dataset(dataname, T, data_dir)

    train_data = PoisonedDataset(
        train_data, trigger_label, mode='train', epsilon=epsilon, device=device,
        pos=pos, trigger_type=type, time_step=T, trigger_size=trigger_size, dataname=dataname)

    test_data_ori = PoisonedDataset(test_data, trigger_label, mode='test', epsilon=0,
                                    device=device, pos=pos, trigger_type=type, time_step=T,
                                    trigger_size=trigger_size, dataname=dataname)

    # TODO: Check if the backdoor is also injected in the same label as the original test data
    test_data_tri = PoisonedDataset(test_data, trigger_label, mode='test', epsilon=1,
                                    device=device, pos=pos, trigger_type=type, time_step=T,
                                    trigger_size=trigger_size, dataname=dataname)

    frame, label = test_data_tri[0]
    play_frame(frame, 'backdoor.gif')

    train_data_loader = DataLoader(
        dataset=train_data,    batch_size=batch_size_train, shuffle=True)
    test_data_ori_loader = DataLoader(
        dataset=test_data_ori, batch_size=batch_size_test, shuffle=True)
    test_data_tri_loader = DataLoader(
        dataset=test_data_tri, batch_size=batch_size_test, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader


def create_dynamic_trigger(size_x, size_y, new_data, height, width, perm, pos, time_step):
    '''
    Creates the dynamic trigger for the backdoor dataset.

    Parameters:
        size_x (int): width of the trigger
        size_y (int): height of the trigger
        new_data (np.array): data to inject the trigger
        height (int): height of the image
        width (int): width of the image
        perm (list): indices of the samples to inject the trigger
        pos (str): position of the trigger. 'top-left', 'top-right', 'bottom-left', 'bottom-right', 'middle', 'random'
        shape (str): shape of the trigger. 'square' or 'random'
        time_step (int): time step of the trigger
        device (torch.device): device to use

    Returns:
        new_data (np.array): data with the injected trigger
    '''

    if pos == 'top-left':
        start_x = size_x + 2
        start_y = size_y + 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'top-right':
        start_x = width - 2
        start_y = size_y + 2

        width_list = [start_x, start_x -
                      size_x - 2, start_x - size_x * 2 - 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'bottom-left':
        start_x = size_x + 2
        start_y = height - 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'bottom-right':
        start_x = height - 2
        start_y = width - 2

        width_list = [start_x, start_x -
                      size_x - 2, start_x - size_x * 2 - 2]
        height_list = [start_y, start_y, start_y]
    elif pos == 'middle':
        start_x = int(width/2) - 2
        start_y = int(height/2) - 2

        width_list = [start_x, start_x +
                      size_x + 2, start_x + size_x * 2 + 2]
        height_list = [start_y, start_y, start_y]

    j = 0
    t = 0

    while t < time_step - 1:
        if j >= len(width_list):
            j = 0

        for x in range(size_x):
            for y in range(size_y):

                new_data[perm, t, 0, height_list[j]-y, width_list[j]-x] = 1
                new_data[perm, t + 1, 1, height_list[j] -
                         y, width_list[j]-x] = 0

        j += 1
        t += 1

    return new_data

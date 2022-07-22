from spikingjelly.clock_driven import functional, surrogate, layer, neuron
import torch
import torch.nn as nn
import torch.nn.functional as F

# Modificar el modelo con esto: https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron/blob/main/codes/models.py


class VotingLayer(nn.Module):
    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class PythonNet(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        conv = []
        conv.extend(PythonNet.conv3x3(2, channels))
        conv.append(nn.MaxPool2d(2, 2))
        for i in range(4):
            conv.extend(PythonNet.conv3x3(channels, channels))
            conv.append(nn.MaxPool2d(2, 2))
        self.conv = nn.Sequential(*conv)
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(channels * 4 * 4, channels * 2 * 2, bias=False),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(0.5),
            nn.Linear(channels * 2 * 2, 110, bias=False),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        )
        self.vote = VotingLayer(10)

    def forward(self, x: torch.Tensor):
        x = x.permute(1, 0, 2, 3, 4)  # [N, T, 2, H, W] -> [T, N, 2, H, W]
        out_spikes = self.vote(self.fc(self.conv(x[0])))
        for t in range(1, x.shape[0]):
            out_spikes += self.vote(self.fc(self.conv(x[t])))
        return out_spikes / x.shape[0]

    @staticmethod
    def conv3x3(in_channels: int, out_channels):
        return [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            neuron.LIFNode(
                tau=2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        ]


def get_models(model_name='dummy', dataname='gesture', load_model=False, path=''):
    '''
    For a given model name, return the model

    Parameters:
        model_name (str): name of the model
        dataname (str): name of the dataset
        load_model (bool): whether to load the model from the path
        path (str): path to the model
    Returns:
        model (nn.Module): the model
    '''
    if dataname == 'gesture':
        num_classes = 11
        n_channels = 128
    elif dataname == 'cifar10':
        num_classes = 10
        n_channels = 34
    elif dataname == 'mnist':
        num_classes = 10
        n_channels = 28
    else:
        raise NotImplementedError(f'{dataname} not implemented')

    if model_name == 'dummy':
        model = PythonNet(n_channels)
        print(model)
    else:
        raise NotImplementedError(f'{model_name} not implemented')

    return model

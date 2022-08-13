import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import layer, surrogate
from spikingjelly.clock_driven.neuron import LIFNode, ParametricLIFNode
from models_old import get_models


def get_model(dataname='gesture', T=16, init_tau=0.02, use_plif=False, use_max_pool=False, detach_reset=False):
    '''
    For a given dataset, return the model according to https://github.com/fangwei123456/Parametric-Leaky-Integrate-and-Fire-Spiking-Neuron

    Parameters:

        dataname (str): name of the dataset.
        T (int): number of time steps.
        init_tau (float): initial tau of the neuron.
        use_plif (bool): whether to use PLIF.
        use_max_pool (bool): whether to use max pooling.
        alpha_learnable (bool): whether to learn the alpha.
        detach_reset (bool): whether to detach the reset.

    Returns:

        model (NeuromorphicNet): the model.
    '''

    if dataname == 'mnist':
        model = NMNISTNet(T, init_tau, use_plif, use_max_pool,
                          detach_reset, 34, 2)
    elif dataname == 'gesture':
        # model = DVS128GestureNet(T, init_tau, use_plif, use_max_pool,
        #                          detach_reset, 128, 5)
        model = get_models()
    elif dataname == 'cifar10':
        # model = CIFAR10DVSNet(T, init_tau, use_plif, use_max_pool,
        #                       detach_reset, 128, 4)
        model = get_models()
    else:
        raise ValueError('Dataset {} is not supported'.format(dataname))

    return model


def create_conv_sequential(in_channels, out_channels, number_layer, init_tau, use_plif, use_max_pool, detach_reset):
    # 首层是in_channels-out_channels
    # 剩余number_layer - 1层都是out_channels-out_channels
    conv = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(
            tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
    ]

    for i in range(number_layer - 1):
        conv.extend([
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(
                tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2)
        ])
    return nn.Sequential(*conv)


def create_2fc(channels, h, w, dpp, class_num, init_tau, use_plif, detach_reset):
    return nn.Sequential(
        nn.Flatten(),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w, channels * h * w // 4, bias=False),
        ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(
            tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
        layer.Dropout(dpp),
        nn.Linear(channels * h * w // 4, class_num * 10, bias=False),
        ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset) if use_plif else LIFNode(
            tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=detach_reset),
    )


class NeuromorphicNet(nn.Module):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset):
        super().__init__()
        self.T = T
        self.init_tau = init_tau
        self.use_plif = use_plif
        self.use_max_pool = use_max_pool
        self.detach_reset = detach_reset

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0
        self.conv = None
        self.fc = None
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        out_spikes_counter = self.boost(
            self.fc(self.conv(x[0])).unsqueeze(1)).squeeze(1)
        for t in range(1, x.shape[0]):
            out_spikes_counter += self.boost(
                self.fc(self.conv(x[t])).unsqueeze(1)).squeeze(1)
        return out_spikes_counter


class NMNISTNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                         detach_reset=detach_reset)
        w = 34
        h = 34  # 原始数据集尺寸
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, use_plif=use_plif, detach_reset=detach_reset)


class CIFAR10DVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                         detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=10,
                             init_tau=init_tau, use_plif=use_plif, detach_reset=detach_reset)


class Interpolate(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwargs)


class ASLDVSNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif,
                         use_max_pool=use_max_pool, detach_reset=detach_reset)
        # input size 256 * 256
        w = 256
        h = 256

        self.conv = nn.Sequential(
            Interpolate(size=256, mode='bilinear'),
        )


class DVS128GestureNet(NeuromorphicNet):
    def __init__(self, T, init_tau, use_plif, use_max_pool, detach_reset, channels, number_layer):
        super().__init__(T=T, init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool,
                         detach_reset=detach_reset)
        w = 128
        h = 128
        self.conv = create_conv_sequential(2, channels, number_layer=number_layer, init_tau=init_tau, use_plif=use_plif,
                                           use_max_pool=use_max_pool, detach_reset=detach_reset)
        self.fc = create_2fc(channels=channels, w=w >> number_layer, h=h >> number_layer, dpp=0.5, class_num=11,
                             init_tau=init_tau, use_plif=use_plif, detach_reset=detach_reset)

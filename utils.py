import torch.nn as nn
from torch import optim


def path_name(args):
    """
    Generate the path name based on th experiment arguments. Use a function for
    that to allow checking the existence of the path from different scripts.

    Parameters:
        args (argparse.Namespace): script arguments.

    Returns:
        path (string): The path used to save our experiments
    """
    path = f'experiments/{args.dataname}_{args.model}_{args.epsilon}_{args.pos}_{args.shape}_{args.trigger_size}_{args.trigger_label}'
    return path


def loss_picker(loss):
    '''
    Select the loss function

    Parameters:
        loss (str): name of the loss function

    Returns:
        loss_function (torch.nn.Module): loss function
    '''
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("Automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum, scheduler, step_size, gamma, T_max):
    '''
    Select the optimizer

    Parameters:
        optimization (str): name of the optimization method
        param (list): model's parameters to optimize
        lr (float): learning rate

    Returns:
        optimizer (torch.optim.Optimizer): optimizer

    '''
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)

    lr_scheduler = None
    if scheduler == 'StepLR':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
    elif scheduler == 'CosALR':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max)

    return optimizer, lr_scheduler

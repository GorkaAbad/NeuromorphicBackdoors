import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import matplotlib as plt
import os
import seaborn as sns
import csv


def path_name(args):
    """
    Generate the path name based on th experiment arguments. Use a function for
    that to allow checking the existence of the path from different scripts.

    Parameters:
        args (argparse.Namespace): script arguments.

    Returns:
        path (string): The path used to save our experiments
    """
    path = f'experiments/{args.dataname}_{args.model}_{args.epsilon}_{args.pos}_{args.trigger_size}_{args.trigger_label}'
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


def optimizer_picker(optimization, param, lr, momentum, T_max):
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
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("Automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_max)

    return optimizer, lr_scheduler


def backdoor_model_trainer(model, criterion, optimizer, epochs, poison_trainloader, clean_testloader,
                           poison_testloader, device, args):
    '''
    Train the model on the backdoored dataset and evaluate the model on the clean and poisoned test dataset
    Parameters:
        model (torch.nn.Module): model
        criterion (torch.nn.modules.loss._Loss): loss function
        optimizer (torch.optim.Optimizer): optimizer
        epochs (int): number of epochs
        poison_trainloader (torch.utils.data.DataLoader): poisoned train loader
        clean_testloader (torch.utils.data.DataLoader): clean test loader
        poison_testloader (torch.utils.data.DataLoader): poisoned test loader
        device (torch.device): device
        args (argparse.Namespace): script arguments.
    Returns:
        list_train_loss (list): list of train loss for each epoch
        list_train_acc (list): list of train accuracy for each epoch
        list_test_loss (list): list of test loss for each epoch
        list_test_acc (list): list of test accuracy for each epoch
        list_test_loss_poison (list): list of test loss for poisoned test dataset
        list_test_acc_poison (list): list of test accuracy for poisoned test dataset
    '''

    if not args.load_model:

        list_train_loss = []
        list_train_acc = []
        list_test_loss = []
        list_test_acc = []
        list_test_loss_backdoor = []
        list_test_acc_backdoor = []

        print(f'\n[!] Training the model for {epochs} epochs')
        print(f'\n[!] Trainset size is {len(poison_trainloader.dataset)},'
              f'Testset size is {len(clean_testloader.dataset)},'
              f'and the poisoned testset size is {len(poison_testloader.dataset)}'
              )

        for epoch in range(epochs):
            train_loss, train_acc = train(
                model, poison_trainloader, optimizer, criterion, device)

            test_loss_clean, test_acc_clean = evaluate(
                model, clean_testloader, criterion, device)

            test_loss_backdoor, test_acc_backdoor = evaluate(
                model, poison_testloader, criterion, device)

            list_train_loss.append(train_loss)
            list_train_acc.append(train_acc)
            list_test_loss.append(test_loss_clean)
            list_test_acc.append(test_acc_clean)
            list_test_loss_backdoor.append(test_loss_backdoor)
            list_test_acc_backdoor.append(test_acc_backdoor)

            print(f'\n[!] Epoch {epoch + 1}/{epochs} '
                  f'Train loss: {train_loss:.4f} '
                  f'Train acc: {train_acc:.4f} '
                  f'Test acc: {test_acc_clean:.4f} '
                  f'Test acc backdoor: {test_acc_backdoor:.4f}'
                  )
    else:
        path = path_name(args)
        data = torch.load(f"{path}/data.pt")
        list_train_loss = data["list_train_loss"]
        list_train_acc = data["list_train_acc"]
        list_test_loss = data["list_test_loss"]
        list_test_acc = data["list_test_acc"]
        list_test_loss_backdoor = data["list_test_loss_backdoor"]
        list_test_acc_backdoor = data["list_test_acc_backdoor"]

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor


def train(model, train_loader, optimizer, criterion, device):
    '''
    Train the model for a single epoch
    Parameters:
        model (torch.nn.Module): model
        train_loader (torch.utils.data.DataLoader): train loader
        optimizer (torch.optim.Optimizer): optimizer
        criterion (torch.nn.modules.loss._Loss): loss function
        device (torch.device): device
    Returns:
        train_loss (float): train loss
        train_acc (float): train accuracy
    '''

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # softmax = nn.Softmax(dim=-1)
        # output = softmax(output)

        if isinstance(criterion, torch.nn.MSELoss):
            loss = criterion(output, target)
        elif isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(output, torch.argmax(target, dim=1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(torch.argmax(target, dim=1)).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / len(train_loader.dataset)

    return train_loss, train_acc


def evaluate(model, test_loader, criterion, device):
    '''
    Evaluate the model on the test set
    Parameters:
        model (torch.nn.Module): model
        test_loader (torch.utils.data.DataLoader): test loader
        criterion (torch.nn.modules.loss._Loss): loss function
        device (torch.device): device
    Returns:
        test_loss (float): test loss
        test_acc (float): test accuracy
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(torch.argmax(target, dim=1)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    return test_loss, test_acc


def plot_accuracy_combined(name, list_train_acc, list_test_acc, list_test_acc_backdoor):
    '''
    Plot the accuracy of the model in the main and backdoor test set
    Parameters:
        name (str): name of the figure
        list_train_acc (list): list of train accuracy for each epoch
        list_test_acc (list): list of test accuracy for each epoch
        list_test_acc_backdoor (list): list of test accuracy for poisoned test dataset
    Returns:
        None
    '''

    sns.set()

    fig, ax = plt.subplots(3, 1)
    fig.suptitle(name)

    ax[0].set_title('Training accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].plot(list_train_acc)

    ax[1].set_title('Test accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].plot(list_test_acc)

    ax[2].set_title('Test accuracy backdoor')
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Accuracy')
    ax[2].plot(list_test_acc_backdoor)

    plt.savefig(f'{name}/accuracy.png',  bbox_inches='tight')
    # Also saving as pdf for using the plot in the paper
    plt.savefig(f'{name}/accuracy.pdf',  bbox_inches='tight')


def save_experiments(args, train_acc, train_loss, test_acc_clean, test_loss_clean, test_acc_backdoor,
                     test_loss_backdoor, model):
    '''
    A function that saves all the info from the experiments. It will create 2 things:
    1. A csv file containing all the info (just text) of the experiments
    2. A folder per experiment containing all the info regarding that experiment.
    NOTE: We also shoud take care of the experiments running more than once.
    The structer of the folder is:
        - experiments
            - experiments.csv
            - {experiment name}
                - args.txt
                - data.pt
    Parameters:
        args (argparse.Namespace): arguments
        train_acc (float): train accuracy
        train_loss (float): train loss
        test_acc_clean (float): test accuracy clean
        test_loss_clean (float): test loss clean
        test_acc_backdoor (float): test accuracy backdoor
        test_loss_backdoor (float): test loss backdoor
        model (torch.nn.Module): model
    Returns:
        None
    '''
    # Create a folder for the experiments, named 'experiments'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Create if not exists a csv file, appending the new info
    path = '{}/experiments.csv'.format(args.save_path)
    header = ['dataname', 'model', 'epsilon', 'pos',
              'shape', 'trigger_size', 'trigger_label',
              'loss', 'optimizer', 'batch_size', 'epochs',
              'train_acc', 'test_acc_clean', 'test_acc_backdoor']

    if not os.path.exists(path):
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    # Append the new info to the csv file
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([args.dataname, args.model, args.epsilon, args.pos,
                         args.shape, args.trigger_size, args.trigger_label,
                         train_loss[-1], args.optimizer, args.batch_size, args.epochs,
                         train_acc[-1], test_acc_clean[-1], test_acc_backdoor[-1]])

    # Create a folder for the experiment, named after the experiment
    path = path_name(args)
    #path = f'experiments/{args.dataname}_{args.model}_{args.epsilon}_{args.pos}_{args.shape}_{args.trigger_size}_{args.trigger_label}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Save the info in a file
    with open(f'{path}/args.txt', 'w') as f:
        f.write(str(args))

    torch.save({
        'args': args,
        'list_train_loss': train_loss,
        'list_train_acc': train_acc,
        'list_test_loss': test_loss_clean,
        'list_test_acc': test_acc_clean,
        'list_test_loss_backdoor': test_loss_backdoor,
        'list_test_acc_backdoor': test_acc_backdoor,
        'model': model
    }, f'{path}/data.pt')

    plot_accuracy_combined(path, train_acc,
                           test_acc_clean, test_acc_backdoor)
    print('Model and results saved successfully!')

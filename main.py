import argparse
from numpy import double
import torch
from models import get_model
from utils import loss_picker, optimizer_picker, backdoor_model_trainer, save_experiments
from poisoned_dataset import create_backdoor_data_loader
from torch.cuda import amp

parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
parser.add_argument('-T', default=16, type=int,
                    help='simulating time-steps')
parser.add_argument('-model', default='dummy', type=str,
                    help='Model name', choices=['dummy'])
parser.add_argument('-trigger', default=0, type=int,
                    help='The index of the trigger label')
parser.add_argument('-polarity', default=0, type=int,
                    help='The polarity of the trigger', choices=[0, 1, 2])
parser.add_argument('-trigger_size', default=0.1,
                    type=float, help='The size of the trigger as the percentage of the image size')
parser.add_argument('-b', default=64, type=int, help='batch size')
parser.add_argument('-b_test', default=64, type=int,
                    help='batch size for test')
parser.add_argument('-epochs', default=64, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-j', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-channels', default=128, type=int,
                    help='channels of Conv2d in SNN')
parser.add_argument('-data_dir', type=str, default='datasets',
                    help='root dir of DVS128 Gesture dataset')
parser.add_argument('-out_dir', type=str, default='datasets',
                    help='root dir for saving logs and checkpoint')
parser.add_argument('-dataname', default='mnist',
                    type=str, help='dataset name', choices=['gesture', 'cifar10', 'mnist'])
parser.add_argument('-resume', type=str,
                    help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true',
                    help='automatic mixed precision training')
parser.add_argument('-opt', type=str, default='sgd',
                    help='use which optimizer. sgd or adam')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-loss', type=str, default='mse', help='loss function')
parser.add_argument('-momentum', default=0.9,
                    type=float, help='momentum for SGD')
parser.add_argument('-step_size', default=32,
                    type=float, help='step_size for StepLR')
parser.add_argument('-T_max', default=32, type=int,
                    help='T_max for CosineAnnealingLR')
parser.add_argument('-epsilon', default=0.1, type=double,
                    help='Percentage of poisoned data')
parser.add_argument('-target', default=0, type=int,
                    help='Index for the target label')
parser.add_argument('-pos', default='top-left', type=str,
                    help='Position of the triggger')
parser.add_argument('-type', default=1, type=int,
                    help='Type of the trigger, 0 static, 1 dynamic')
parser.add_argument('-seed', default=42, type=int,
                    help='The random seed value')
parser.add_argument('-load_model', action='store_true',
                    help='load model from checkpoint')
parser.add_argument('-trigger_label', default=0, type=int,
                    help='The index of the trigger label')
parser.add_argument('-tau', default=2.0, type=float,
                    help='Tau for the LIF node')
parser.add_argument('-use_plif', action='store_true',
                    default=False, help='Use PLIF')
parser.add_argument('-use_max_pool', action='store_true',
                    default=False, help='Use MaxPool')
parser.add_argument('-detach_reset', action='store_true',
                    default=False, help='Detach reset')
parser.add_argument('-save_path', type=str, default='.',
                    help='Path to save the model')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get the model, loss and optimizer
    model = get_model(args.dataname, args.T, args.tau, args.use_plif,
                      args.use_max_pool, args.detach_reset)

    model.to(device)

    criterion = loss_picker(args.loss)

    optimizer, lr_scheduler = optimizer_picker(
        args.opt, model.parameters(), args.lr, args.momentum, args.T_max)

    poison_trainloader, clean_testloader, poison_testloader = create_backdoor_data_loader(
        args.dataname, args.trigger, args.epsilon, args.pos, args.type, args.trigger_size,
        args.b, args.b_test, args.T, device, args.data_dir, args.polarity)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    # Train the model
    list_train_loss, list_train_acc, list_test_loss, list_test_acc, list_test_loss_backdoor, list_test_acc_backdoor = backdoor_model_trainer(
        model, criterion, optimizer, args.epochs, poison_trainloader, clean_testloader, poison_testloader,
        device, args, scaler)

    # Save the results
    save_experiments(args, list_train_acc, list_train_loss, list_test_acc, list_test_loss, list_test_acc_backdoor,
                     list_test_loss_backdoor, model)


if __name__ == '__main__':
    main()

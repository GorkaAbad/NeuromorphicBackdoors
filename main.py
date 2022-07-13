import argparse
import torch

from models import get_models
from utils import loss_picker, optimizer_picker
from poisoned_dataset import create_backdoor_data_loader

parser = argparse.ArgumentParser(description='Classify DVS128 Gesture')
parser.add_argument('-T', default=16, type=int,
                    help='simulating time-steps')
parser.add_argument('-model', default='dummy', type=str, help='model name')
parser.add_argument('-device', default='cpu:0', help='device')
parser.add_argument('-trigger', default=0, type=int,
                    help='The index of the trigger label')
parser.add_argument('-polarity', default=0, type=int,
                    help='The polarity of the trigger')
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
parser.add_argument('-data_dir', type=str, default='DVS Gesture dataset',
                    help='root dir of DVS128 Gesture dataset')
parser.add_argument('-out_dir', type=str, default='DVS Gesture dataset',
                    help='root dir for saving logs and checkpoint')
parser.add_argument('-dataname', default='cifar10',
                    type=str, help='dataset name')
parser.add_argument('-resume', type=str,
                    help='resume from the checkpoint path')
parser.add_argument('-amp', action='store_true',
                    help='automatic mixed precision training')
parser.add_argument('-cupy', action='store_true',
                    help='use CUDA neuron and multi-step forward mode')

parser.add_argument('-opt', type=str, default='SGD',
                    help='use which optimizer. SDG or Adam')
parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
parser.add_argument('-loss', type=str, default='mse', help='loss function')
parser.add_argument('-momentum', default=0.9,
                    type=float, help='momentum for SGD')
parser.add_argument('-lr_scheduler', default='CosALR',
                    type=str, help='use which schedule. StepLR or CosALR')
parser.add_argument('-step_size', default=32,
                    type=float, help='step_size for StepLR')
parser.add_argument('-gamma', default=0.1, type=float,
                    help='gamma for StepLR')
parser.add_argument('-T_max', default=32, type=int,
                    help='T_max for CosineAnnealingLR')
parser.add_argument('-epsilon', default=0.1, type=float,
                    help='Percentage of poisoned data')
parser.add_argument('-target', default=0, type=int,
                    help='Index for the target label')
parser.add_argument('-pos', default='top-left', type=str,
                    help='Position of the triggger')
parser.add_argument('-type', default=1, type=int,
                    help='Type of the trigger, 0 static, 1 dynamic')
parser.add_argument('-seed', default=42, type=int,
                    help='The random seed value')

args = parser.parse_args()


def main():
    torch.manual_seed(args.seed)
    # path = path_name(args)
    device = torch.device(args.device)

    # Get the model, loss and optimizer
    model = get_models(args.model,
                       dataname=args.dataname, load_model=False,
                       path=args.data_dir)

    criterion = loss_picker(args.loss)
    optimizer, lr_scheduler = optimizer_picker(
        args.opt, model.parameters(), args.lr, args.momentum, args.lr_scheduler,
        args.step_size, args.gamma, args.T_max)

    poison_trainloader, clean_testloader, poison_testloader = create_backdoor_data_loader(
        args.dataname, args.trigger, args.epsilon, args.pos, args.type, args.trigger_size,
        args.b, args.b_test, args.T, device)


if __name__ == '__main__':
    main()

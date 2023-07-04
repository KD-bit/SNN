import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from model import *
from layers import *
from tensorboardX import SummaryWriter

batch_size = 64
steps = STEPS
log_interval = 10
lr = 1e-3
epochs = 100

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

'''train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('E:\learning\毕设\snntorch\data\mnist\MNIST', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.RandomHorizontalFlip(),
                         transforms.RandomCrop(28, padding=4),
                         transforms.ToTensor(),
                         transforms.Normalize(0, 1)
                     ])),
    batch_size=batch_size, shuffle=True)'''

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('E:\learning\毕设\snntorch\data\mnist\MNIST', train=True, download=True,
                     transform=transform), batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('E:\learning\毕设\snntorch\data\mnist\MNIST', train=False,
                     transform=transform), batch_size=batch_size, shuffle=True)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # necessary for general dataset: broadcast input
        data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
        data = data.permute(1, 2, 3, 4, 0)

        output = model(data)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data / steps), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    isEval = False
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            data, _ = torch.broadcast_tensors(data, torch.zeros((steps,) + data.shape))
            data = data.permute(1, 2, 3, 4, 0)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    acc = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return acc

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 35))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = MNISTNet().to(device)
    acc_record = []

    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0

    checkpoint_path = './ckpt_model/tdBN_MNIST.pkl'
    '''if checkpoint_path is not None and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print("start_epoch:{}".format(start_epoch))
        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))'''

    for epoch in range(start_epoch, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader, epoch)
        acc_record.append(acc)
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'acc_record':acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './ckpt_model/MNIST.pkl')

if __name__ == '__main__':
    main()





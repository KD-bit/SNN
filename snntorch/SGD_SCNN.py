import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

#print(torch.cuda.is_available())
names = 'SGD_avgpool_SCNN'

acc_record = list([])

# dataloader arguments
batch_size = 128
data_path='/data/mnist'
subset=10

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# reduce datasets by 10x to speed up training
utils.data_subset(mnist_train, subset)
utils.data_subset(mnist_test, subset)

# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)

beta = 0.9
num_steps = 50

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

#  Initialize Network
net = nn.Sequential(nn.Conv2d(1, 12, 5),        #[128, 1, 28, 28] -> [128, 12, 24, 24]
                    nn.AvgPool2d(2),            #[128, 12, 24, 24] -> [128, 12, 12, 12]
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(12, 64, 5),       #[128, 12, 12, 12] -> [128, 64, 8, 8]
                    nn.AvgPool2d(2),            #[128, 64, 8, 8] -> [128, 64, 4, 4]
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),               #[128, 64, 4, 4] -> [128, 64*4*4]
                    nn.Linear(64*4*4, 10),      #[128, 10]
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

data, targets = next(iter(train_loader))
data = data.to(device)
targets = targets.to(device)

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)

acc = SF.accuracy_rate(spk_rec, targets)

#total accuracy
def batch_accuracy(train_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

        train_loader = iter(train_loader)
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec, _ = forward_pass(net, num_steps, data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc / total

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2, betas=(0.9, 0.999))
num_epochs = 1
test_acc_hist = []

# training loop
for epoch in range(num_epochs):
    avg_loss = backprop.BPTT(net, train_loader, optimizer=optimizer, criterion=loss_fn,
                             num_steps=num_steps, time_var=False, device=device)

    print(f"Epoch {epoch}, Train Loss: {avg_loss.item():.2f}")

    # Test set accuracy
    test_acc = batch_accuracy(test_loader, net, num_steps)
    test_acc_hist.append(test_acc)

    print(f"Epoch {epoch}, Test Acc: {test_acc * 100:.2f}%\n")
    acc_record.append(test_acc)
    optimizer = lr_scheduler(optimizer, epoch)
    if (epoch+1) % 5 == 0:
        print(acc)
        print('Saving..')
        state = {
            #'net': snn.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'acc_record': acc_record,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # torch.save(state, './checkpoint/ckpt_state_dict' + names + '.t7')
        torch.save(state, './checkpoint/ckpt_state_dict' + names + '.pkl')

model = torch.load('E:\learning\毕设\snntorch\checkpoint\ckptSGD_avgpool_SCNN.pkl')

acc_record = list(model['acc_record'])
plt.plot(acc_record)
plt.title('SCNN')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

num = 0
total_acc = 0
for i in range(len(acc_record)):
    num += 1
    total_acc += acc_record[i]

total_acc = (total_acc / num) * 100

print(f"Total correctly classified test set images: %.2f" % total_acc)

spk_rec, mem_rec = forward_pass(net, num_steps, data)

from IPython.display import HTML

idx = 0

fig, ax = plt.subplots(facecolor='w', figsize=(12, 7))
labels=['0', '1', '2', '3', '4', '5', '6', '7', '8','9']
print(f"The target label is: {targets[idx]}")

# plt.rcParams['animation.ffmpeg_path'] = 'C:\\path\\to\\your\\ffmpeg.exe'

#  Plot spike count histogram
anim = splt.spike_count(spk_rec[:, idx].detach().cpu(), fig, ax, labels=labels,
                        animate=True, interpolate=4)    #spk_rec[:, idx] [50, 10]the first parameter is time step,
                                                        #the second parameter is the counts of labels

HTML(anim.to_html5_video())
anim.save("spike_bar2.mp4")



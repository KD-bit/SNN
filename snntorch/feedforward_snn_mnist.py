import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import functional as SF
from snntorch import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

import os

#dataloader arguments
batch_size = 128
data_path = '/data/mnist'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

# Network Architecture
num_inputs = 28 * 28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

#Define network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)


    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        mem2_rec = []
        spk1_rec = []
        spk2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])  # post-synaptic current <-- spk_in x weight
            spk1, mem1 = self.lif1(cur1, mem1)  # mem[t+1] <--post-syn current + decayed membrane
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            mem2_rec.append(mem2)
            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        # convert lists to tensors
        mem2_rec = torch.stack(mem2_rec)
        spk1_rec = torch.stack(spk1_rec)
        spk2_rec = torch.stack(spk2_rec)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

net = Net().to(device)

'''net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 1000),
                    snn.Leaky(beta=beta),
                    nn.Linear(1000, 10),
                    snn.Leaky(beta=beta, output=True)).to(device)'''

#define optimizer and criterion
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

num_epochs = 100
loss_hist = []
test_loss_hist = []
test_acc_hist = []
names = 'feedforward_snn'

#Outer loop
for epoch in range(num_epochs):
    running_loss = 0
    train_batch = iter(train_loader)

    for data, targets in train_batch:
        data = data.to(device)  # [128, 1, 28, 28]
        targets = targets.to(device)  # [128]

        # forward pass
        net.train()
        spk_rec, mem_rec = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        running_loss = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            running_loss += loss(mem_rec[step], targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(running_loss.item())

    print(f"epoch: {epoch + 1}, loss: {running_loss}" )

    correct = 0
    total = 0
    test_batch = iter(test_loader)

    with torch.no_grad():
        net.eval()
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            test_spk, _ = net(data.view(data.size(0), -1))

            # calculate total accuracy
            _, predicted = test_spk.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_acc_hist.append(correct / total)

    state = {
        'net': net.state_dict(),
        'acc': correct / total,
        'epoch': epoch,
        'acc_record': test_acc_hist,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_state_dict' + names + '_5layers.pkl')

    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")


plt.plot(test_acc_hist)
plt.title('feedforward_snn_5layers')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

with torch.no_grad():
    net.eval()
    acc = 0
    total = 0
    test_loader = iter(test_loader)

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)
        spk, _ = net(data.view(batch_size, -1))

        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"total_acc:{correct / total}")

model = torch.load('E:\learning\毕设\snntorch\checkpoint\ckptfeedforward_snn.pkl')

acc_record = list(model['acc_record'])













import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import math
import os

def xavier_init(m):
    if type(m) == nn.Conv2d:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    if type(m) == nn.Linear:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        N = 300
        self.inputShape = (N, 2)
        self.cnn1 = nn.Sequential(  # input shape (1, 300, 2)
            nn.Conv2d(1,  # input channel
                      32,  # output channel
                      kernel_size=(1,2),  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(),
            nn.BatchNorm2d(32))
        self.cnn2 = nn.Sequential(
            nn.Conv2d(32,  # input channel
                      32,  # output channel
                      kernel_size=(1, 2),  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            )
        self.pool = nn.MaxPool2d(kernel_size=(N, 1), stride=1)

        self.featureSize = self.featureSize()
        self.layer2 = nn.Sequential(
            nn.Linear(self.featureSize, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3))  # inputWdith / 2
        # add a fully connected layer
        # width = int(inputWidth / 4) + 1
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3))  # inputWdith / 2

        self.prediction = nn.Linear(128, 1)
        self.apply(xavier_init)

    def forward(self, x):
        x = self.pool(self.cnn2(self.cnn1(x)))
        x = x.reshape(x.size(0),-1)

        return self.prediction(self.layer3(self.layer2(x)))

    def featureSize(self):
        return self.pool(self.cnn2(self.cnn1(torch.zeros(1, 1, *self.inputShape)))).view(1, -1).size(1)



net = ConvNet()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_func = torch.nn.MSELoss()

if torch.cuda.is_available():
    net.cuda()
    device = torch.device("cuda")
with np.load('opfilep300.npz') as data:
    opSampleAll = data['a']
    xyzSampleAll = data['b']

samples = opSampleAll.shape[0]

Rgmax = np.max(opSampleAll[:,0])
Rgmin = np.min(opSampleAll[:,0])
opSampleAll[:, 0] = (opSampleAll[:, 0] - Rgmin)/(Rgmax - Rgmin)
xyzSampleAll /= 30.0

XAll = np.expand_dims(xyzSampleAll, axis = 1)
YAll = opSampleAll[:, 0]

dataSet = Data.TensorDataset(torch.from_numpy(XAll.astype(np.float32)), torch.from_numpy(YAll.astype(np.float32)))
loader = Data.DataLoader(
    dataset=dataSet,            # torch TensorDataset format
    batch_size=32,              # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

for epoch in range(5000):
    for (batch_x, batch_y) in loader:

        if torch.cuda.is_available():
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

        prediction = net(batch_x)  # input x and predict based on x

        loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients



    print("loss at epoch" + str(epoch) + " is:" + str(loss.data))
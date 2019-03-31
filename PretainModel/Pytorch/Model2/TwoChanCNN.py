
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import math
import os
import json
from BDModel.BDQuadModelEnv import *

def xavier_init(m):
    if type(m) == nn.Conv2d:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    if type(m) == nn.Conv1d:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:3])
        fan_out = np.prod(weight_shape[2:3]) * weight_shape[0]
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
class TwoChanConvNet(nn.Module):
    def __init__(self, posInputWidth, oriInputWidth, numActions):
        super(TwoChanConvNet, self).__init__()
        self.posInputShape = (posInputWidth, posInputWidth)
        self.oriInputShape = oriInputWidth
        self.posLayer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv2d(1,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2

        self.posLayer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # inputWdith / 2

        self.oriLayer1 = nn.Sequential(  # input shape (1, inputWdith, inputWdith)
            nn.Conv1d(1,  # input channel
                      32,  # output channel
                      kernel_size=2,  # filter size
                      stride=1,
                      padding=1),
            # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))  # inputWdith / 2

        self.oriLayer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))  # inputWdith / 2

        self.posFeatureSize = self.posFeatureSize()
        self.oriFeatureSize = self.oriFeatureSize()

        self.fc1 = nn.Sequential(
            nn.Linear(self.posFeatureSize + self.oriFeatureSize , 128),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(128 , 128),
            nn.ReLU())
        self.prediction = nn.Linear(128, numActions)
        self.apply(xavier_init)


    def forward(self, state):
        x = state['position']
        y = state['orientation']
        xout = self.posLayer2(self.posLayer1(x))
        yout = self.oriLayer2(self.oriLayer1(y))

        xout = xout.reshape(xout.size(0), -1)
        yout = yout.reshape(yout.size(0), -1)
        # mask xout for test
        #xout.fill_(0)

        out = torch.cat((xout, yout), 1)
        out = self.fc2(self.fc1(out))
        out = self.prediction(out)
        return out

    def posFeatureSize(self):
        return self.posLayer2(self.posLayer1(torch.zeros(1, 1, *self.posInputShape))).view(1, -1).size(1)

    def oriFeatureSize(self):
        return self.oriLayer2(self.oriLayer1(torch.zeros(1, 1, self.oriInputShape))).view(1, -1).size(1)


def stateProcessor(state, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, state)), device=device, dtype=torch.uint8)

    senorList = [item['position'] for item in state if item is not None]
    targetList = [item['orientation'] for item in state if item is not None]
    nonFinalState = {'sensor': torch.tensor(senorList, dtype=torch.float32, device=device),
              'target': torch.tensor(targetList, dtype=torch.float32, device=device)}
    return nonFinalState, nonFinalMask

net = TwoChanConvNet(50, 72, 2)

if torch.cuda.is_available():
    net.cuda()
    device = torch.device("cuda")

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

configIdx = range(0, 200, 5)
#configIdx = [90]
env = BDQuadModelEnv_v1(config, 1)

with np.load('opfilep300.npz') as data:
    opSampleAll = data['a']
    xyzSampleAll = data['b']

samples = opSampleAll.shape[0]
Rgmax = np.max(opSampleAll[:,0])
Rgmin = np.min(opSampleAll[:,0])
opSampleAll[:, 0] = (opSampleAll[:, 0] - Rgmin)/(Rgmax - Rgmin)

XAll = np.expand_dims(xyzSampleAll, axis = 1)
YAll = opSampleAll[:, 0:2]

posSensorList = []
oriSensorList = []

sampleToTrain = 4000
for i in range(sampleToTrain+6000):
    xy = np.squeeze(XAll[i,:,:])
    sensorMat = BDQuadModelEnv_v1.getSenorInfo(xy)
    posSensorList.append(np.expand_dims(sensorMat,0))
    oriSensorList.append(np.expand_dims(BDQuadModelEnv_v1.getOrientationHist(xy),0))



optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_func = torch.nn.MSELoss()

miniBatchSize = 128
testIdx = range(sampleToTrain, 6000, 1)
for batch in range(5000):
    idx = np.random.choice(sampleToTrain, miniBatchSize)
    batch_x_pos = [posSensorList[i] for i in idx]
    batch_x_ori = [oriSensorList[i] for i in idx]
    batch_y = YAll[idx, :]
    batch_state = {'position': torch.tensor(batch_x_pos, dtype=torch.float32, device=device),
            'orientation': torch.tensor(batch_x_ori, dtype=torch.float32, device=device)}
    batch_y = torch.tensor(batch_y, dtype=torch.float32, device=device)

    prediction = net(batch_state)  # input x and predict based on x

    loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)

    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    if batch % 100 == 0:

        print("loss at batch" + str(batch) + " is:" + str(loss.data))

batch_x_pos = [posSensorList[i] for i in testIdx]
batch_x_ori = [oriSensorList[i] for i in testIdx]
batch_y = YAll[testIdx, :]
batch_state = {'position': torch.tensor(batch_x_pos, dtype=torch.float32, device=device),
        'orientation': torch.tensor(batch_x_ori, dtype=torch.float32, device=device)}
batch_y = torch.tensor(batch_y, dtype=torch.float32, device=device)

prediction = net(batch_state)  # input x and predict based on x

loss = loss_func(prediction, batch_y)  # must be (1. nn output, 2. target)

np.savetxt('testY.txt', batch_y.detach().cpu().numpy())
np.savetxt('prediction.txt', prediction.detach().cpu().numpy())
torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, 'net.pt')

import numpy as np
from Agents.DQN.DQNSyn import DQNSynAgent
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


def stateProcessor(partConfig, device = 'cpu'):
    # given a list a dictions like { 'sensor': np.array, 'target': np.array}
    # we want to get a diction like {'sensor': list of torch tensor, 'target': list of torch tensor}
    nonFinalMask = torch.tensor(tuple(map(lambda s: s is not None, partConfig)), device=device, dtype=torch.uint8)

    posSensorList = []
    oriSensorList = []
    for xy in partConfig:
        if xy is not None:
            xy = np.array(xy)
            sensorMat = BDQuadModelEnv_v1.getSenorInfo(xy)
            posSensorList.append(np.expand_dims(sensorMat, 0))
            oriSensorList.append(np.expand_dims(BDQuadModelEnv_v1.getOrientationHist(xy), 0))

    nonFinalState = {'position': torch.tensor(posSensorList, dtype=torch.float32, device=device),
              'orientation': torch.tensor(oriSensorList, dtype=torch.float32, device=device)}
    return nonFinalState, nonFinalMask



from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv

configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

configIdx = range(0, 200, 5)



# we need a wrapper
def make_env(config, i):
    def _thunk():
        env = BDQuadModelEnv_v1(config, i)

        return env
    return _thunk
numWorkers = config['numWorkers']

envs = [make_env(config,  i) for i in range(numWorkers)]
if numWorkers > 1:
    envs = SubprocVecEnv(envs)
else:
    envs = SubprocVecEnv(envs)


N_A = 4

policyNet = TwoChanConvNet(50, 72, N_A)

checkpoint = torch.load('net.pt')
print(checkpoint['model_state_dict'].keys())
pretrained_dict = checkpoint['model_state_dict']
model_dict = policyNet.state_dict()

# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['prediction.weight', 'prediction.bias']}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
policyNet.load_state_dict(model_dict)



configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

targetNet = deepcopy(policyNet)

optimizer = optim.Adam(policyNet.parameters(), lr=config['learningRate'])


agent = DQNSynAgent(config, policyNet, targetNet, envs, optimizer, torch.nn.MSELoss(reduction='none'), N_A, stateProcessor=stateProcessor)


agent.train()

print('done Training')

testFlag = True
if testFlag:
    config['BDModeloutputFileTag'] = 'trajTest_seed_'
    config['BDModelOutputFlag'] = 2
    envs = [make_env(config, i) for i in range(numWorkers)]
    envs = SubprocVecEnv(envs)

    state = envs.reset()
    stepCountList = np.zeros(numWorkers)
    epIdx = 0
    for i in range(100):
        actions = agent.select_action(agent.policyNet,state, -0.1)
    # note that in vector envs: if one env finishes, it will automatically reset and start a new episode
        state, reward, done, info = envs.step(actions)
        if np.any(done):
            idx = np.where(done == True)
            # adjust for states to avoid None in states
            stepCountDone = stepCountList[idx]
            stepCountList[idx] = 0.0
            epIdx += len(idx[0])
            print('episode', epIdx)
            print("done in step count:")
            print(stepCountDone)

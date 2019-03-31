
import numpy as np

import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as Data
import math
import os
import json
from BDModel.BDQuadModelEnv import *
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

numWorkers = 4

envs = [make_env(config,  i) for i in range(numWorkers)]
if numWorkers > 1:
    envs = SubprocVecEnv(envs)
else:
    envs = SubprocVecEnv(envs)


N_A = 4


state = envs.reset()

for i in range(10):
    actionList = []
    for _ in range(numWorkers):
        action = random.randint(0, N_A - 1)
        actionList.append(action)

    # note that in vector envs: if one env finishes, it will automatically reset and start a new episode
    state, reward, done, _ = envs.step(actionList)
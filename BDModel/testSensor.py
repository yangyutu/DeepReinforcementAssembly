#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:42:05 2019

@author: yangyutu123
"""

from BDQuadModelEnv import *
import numpy as np
import matplotlib.pyplot as plt
import json

partConfig = np.genfromtxt('HCPconfig.txt')

refHist, edges = computeOrienHist(partConfig)

alignHist = alginOrienHist(refHist, refHist)

plt.close('all')
plt.figure(1)
plt.plot(alignHist)


configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

env = BDQuadModelEnv_v1(config, 1)

sensorMat = env.getSenorInfo(partConfig)

plt.figure(2)
plt.imshow(sensorMat)
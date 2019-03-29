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
configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)
opData = np.genfromtxt('trajAction3op0.dat')
opData = opData[:,[1,3]]
xyzData = np.genfromtxt('trajAction3xyz_0.dat')

xyzData = np.reshape(xyzData[:,1:3], (-1, 300, 2))

configIdx = range(0, 200, 5)
#configIdx = [90]
env = BDQuadModelEnv_v1(config, 1)
plt.close('all')
for i in configIdx:
    xy = np.squeeze(xyzData[i,:,:])
    fig, ax = plt.subplots(2,2, figsize=(20,20))
    ax[0,0].scatter(xy[:,0], xy[:,1])
    ax[1, 0].plot(env.refHist)
    ax[1, 0].plot(env.getOrientationHist(xy))
    ax[1, 0].legend(['ref HCP', 'actual'])
    sensorMat = env.getSenorInfo(xy)
    
    ax[0, 1].imshow(sensorMat)
    
    plt.savefig('config'+str(i)+'.png')
    plt.close()

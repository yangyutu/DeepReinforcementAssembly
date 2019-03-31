#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:53:48 2019

@author: yangyutu123
"""

import numpy as np
import BDModel.Model_QuadrupoleBD as BDModel
import random
import gym
import os

from copy import deepcopy
import math
from sklearn.neighbors import NearestNeighbors

def alginOrienHist(refHist, inputHist):
    nbins = len(refHist)
    candidateHist = [np.roll(inputHist, i) for i in range(nbins)]
    distance = list(map(lambda hist: np.linalg.norm(hist - refHist, ord=1), candidateHist))
    #distance = map(lambda hist: np.linalg.norm(hist - refHist, ord=1), candidateHist)
    minIdx = np.argmin(distance)
    return candidateHist[minIdx]



def computeOrienHist(partConfig):
    nbs = NearestNeighbors(n_neighbors=7,algorithm='ball_tree').fit(partConfig)
    distances, indices = nbs.kneighbors(partConfig)
    angle = []
    for i in range(partConfig.shape[0]):
        x0 = partConfig[i,0]
        y0 = partConfig[i,1]    
        for nb_idx, dist in zip(indices[i,1:],distances[i,1:]):
            if dist < 2.85:
                x1 = partConfig[nb_idx,0]
                y1 = partConfig[nb_idx,1]
                angle.append(math.atan2(y1-y0,x1-x0)*180.0/math.pi)

    angle = np.array(angle)
    delAngle = 5
    angle = angle + 180
    for i, a in enumerate(angle):
        if a > 360 - delAngle/2:
            angle[i] = a - 360
    #angleIdx = (int(np.floor(angle + 180 + 0.5*delAngle) / delAngle))*delAngle + 0.5*delAngle
    descriptor,edges = np.histogram(angle,bins=int(360/delAngle),range=(- delAngle/2, 360 - delAngle/2.0))
    return descriptor, edges
    

class BDQuadModelEnv_v0(gym.Env):
    N = 300
    def __init__(self, config, randomSeed = 1):
        self.config = config
        self.outputFlag = 2
        if 'BDModelOutputFlag' in self.config:
            self.outputFlag = self.config['BDModelOutputFlag']

        self.dirName = 'Traj/'
        if not os.path.exists(self.dirName):
            os.makedirs(self.dirName)

        self.outputFileTag = 'trajTrain_seed_'
        if 'BDModeloutputFileTag' in self.config:
            self.outputFileTag = self.config['BDModeloutputFileTag']

        self.model = BDModel.Model_QuadrupoleBD(self.outputFileTag+str(randomSeed), self.outputFlag, randomSeed)
        self.nbActions = 4
        self.stateDim = 600

        self.initConfigFileTag = self.config['BDModelConfigFile']
        self.initConfigFileRange = self.config['BDModelConfigFileRange']
        
        self.controlStep = 1
        if 'BDModelControlStep' in self.config:
            self.controlStep = self.config['BDModelControlStep']
        
        self.episodeCount = 0

        self.episodeEndStep = 500
        if 'BDModelEpisodeEndStep' in self.config:
            self.episodeEndStep = self.config['BDModelEpisodeEndStep']
        
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        self.stepCount = 0

        self.infoDict = {'Psi6': 0.0, 'C6': 0.0, 'Rg': 0.0, 'action': 0.0, 'lambda': 0.0, 'reset': False, 'endBeforeDone': False}

    def reset(self):
        self.infoDict['reset'] = True
        self.maxPsi6 = 0.0
        self.stepCount = 0
        fileCount = self.episodeCount % self.initConfigFileRange
        initFile = self.initConfigFileTag + str(fileCount) + '.txt'
        self.model.setInitialConfigFile(initFile)
        self.model.createInitialState()
        partConfig = self.model.getParticleConfig()
        partConfig = np.reshape(partConfig, (BDQuadModelEnv_v0.N, 2))
        self.episodeCount += 1
        info = self.model.getInfo()
        print('reset info')
        print(info)
        
        return partConfig
        
    def step(self, action):
        if self.stepCount == 0:
            self.infoDict['reset'] = True
        else:
            self.infoDict['reset'] = False
        self.infoDict['endBeforeDone'] = False
        for i in range(self.controlStep):
            self.model.run(action)
            self.stepCount += 1

        self.partConfig = self.model.getParticleConfig()
        partConfig = np.reshape(self.partConfig, (BDQuadModelEnv_v0.N, 2))
        info = self.model.getInfo()
        info = {'Psi6':info[0], 'C6':info[1], 'Rg': info[2], 'action': info[3], 'lambda': info[4]}
        self.infoDict.update(info)

        done, reward = self.calReward(info)



        return partConfig, reward, done, self.infoDict.copy()
    
    def calReward(self, info):
        psi6 = self.infoDict['Psi6']
        rg = self.infoDict['Rg']
        done = False
        
        if psi6 > 0.95 and rg < 13.5:
            done = True
            reward = 1
#        elif self.maxPsi6 < psi6:
#            reward = psi6 - self.maxPsi6
#            self.maxPsi6 = psi6
        else:
            reward = 0.0


        if self.stepCount > self.episodeEndStep:
            done = True
            self.infoDict['endBeforeDone'] = True
               
        return done, reward
        
    def close(self):
        pass

    def seed(self):
        pass        



class BDQuadModelEnv_v1(BDQuadModelEnv_v0):
    offSet = 0
    sensorMatSize = 0
    sensorPixelSize = 0
    refConfig = None
    refHist = None

    def __init__(self, config, randomSeed):

        super(BDQuadModelEnv_v1, self).__init__(config, randomSeed)
               
        BDQuadModelEnv_v1.offSet = self.config['BDModelSenorOffset']
        BDQuadModelEnv_v1.sensorMatSize = self.config['BDModelSenorMatSize']
        BDQuadModelEnv_v1.sensorPixelSize = self.config['BDModelSensorPixelSize']
        
        BDQuadModelEnv_v1.refConfig = np.genfromtxt('HCPconfig.txt')
        BDQuadModelEnv_v1.refHist, _ = computeOrienHist(self.refConfig)
        BDQuadModelEnv_v1.refHist = self.refHist / self.N
              

    @classmethod
    def getSenorInfo(cls, partConfig):
    
        x = partConfig[:, 0] + BDQuadModelEnv_v1.offSet
        y = partConfig[:, 1] + BDQuadModelEnv_v1.offSet
        

        
        xIdx = np.floor(x / BDQuadModelEnv_v1.sensorPixelSize).astype(np.int)
        yIdx = np.floor(y / BDQuadModelEnv_v1.sensorPixelSize).astype(np.int)


        xIdx[xIdx < 0] = 0
        yIdx[yIdx < 0] = 0
        xIdx[xIdx >= BDQuadModelEnv_v1.sensorMatSize] = BDQuadModelEnv_v1.sensorMatSize - 1
        yIdx[yIdx >= BDQuadModelEnv_v1.sensorMatSize] = BDQuadModelEnv_v1.sensorMatSize - 1
        sensorMat = np.zeros((BDQuadModelEnv_v1.sensorMatSize, BDQuadModelEnv_v1.sensorMatSize), dtype=np.int32)
        sensorMat[xIdx, yIdx] = 1
        
        return sensorMat
    @classmethod
    def getOrientationHist(cls, partConfig):

        hist, _ = computeOrienHist(partConfig)
        hist = hist / BDQuadModelEnv_v1.N
        return alginOrienHist(BDQuadModelEnv_v1.refHist, hist)
    
     


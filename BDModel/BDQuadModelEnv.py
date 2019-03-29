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
    def __init__(self, config, randomSeed):
        self.config = config
        self.model = BDModel.Model_QuadrupoleBD('trajTest', 0, randomSeed)
        self.nbActions = 4
        self.stateDim = 600
        self.N = 300
        self.initConfigFileTag = self.config['BDModelConfigFile']
        self.initConfigFileRange = self.config['BDModelConfigFileRange']
        
        self.controlStep = 1
        if 'BDModelControlStep' in self.config:
            self.controlStep = self.config['BDModelControlStep']
        
        self.episodeCount = 0
        
        # import parameter for vector env
        self.viewer = None
        self.steps_beyond_done = None
        
    def reset(self):
        self.maxPsi6 = 0.0
        fileCount = self.episodeCount % self.initConfigFileRange
        initFile = self.initConfigFileTag + str(fileCount) + '.txt'
        self.model.setInitialConfigFile(initFile)
        self.model.createInitialState()
        partConfig = self.model.getParticleConfig()
        
        self.episodeCount += 1
        
        return partConfig
        
    def step(self, action):
        
        for i in range(self.controlStep):
            self.model.run(action)
        
        partConfig = self.model.getParticleConfig()
        info = self.model.getInfo()
        done, reward = self.calReward(info)
        return partConfig, reward, done, info
    
    def calReward(self, info):
        psi6 = info[0]
        
        done = False
        
        if psi6 > 0.95:
            done = True
            reward = 5
        elif self.maxPsi6 < psi6:
            reward = psi6 - self.maxPsi6            
            self.maxPsi6 = psi6
        else:
            reward = -0.1
               
        return done, reward
        
    def close(self):
        pass

    def seed(self):
        pass        



class BDQuadModelEnv_v1(BDQuadModelEnv_v0):
    def __init__(self, config, randomSeed):
        super(BDQuadModelEnv_v1, self).__init__(config, randomSeed)
               
        self.offSet = self.config['BDModelSenorOffset']
        self.sensorMatSize = self.config['BDModelSenorMatSize']
        self.sensorPixelSize = self.config['BDModelSensorPixelSize']
        
        self.refConfig = np.genfromtxt('HCPconfig.txt')
        self.refHist, _ = computeOrienHist(self.refConfig)
        self.refHist = self.refHist / self.N
              
    def step(self, action):
        self.model.run(action)
        self.partConfig = self.model.getParticleConfig()
        info = self.model.getInfo()
        done, reward = self.calReward(info)
        return deepcopy(self.partConfig), reward, done, info
    
    def calReward(self, info):
        psi6 = info[0]
        
        done = False
        
        if psi6 > 0.95:
            done = True
            reward = 5
        elif self.maxPsi6 < psi6:
            reward = psi6 - self.maxPsi6            
            self.maxPsi6 = psi6
        else:
            reward = -0.1
               
        return done, reward
    
    def getSenorInfo(self, partConfig):
    
        x = partConfig[:, 0] + self.offSet
        y = partConfig[:, 1] + self.offSet
        

        
        xIdx = np.floor(x / self.sensorPixelSize).astype(np.int)
        yIdx = np.floor(y / self.sensorPixelSize).astype(np.int)


        xIdx[xIdx < 0] = 0
        yIdx[yIdx < 0] = 0
        xIdx[xIdx >= self.sensorMatSize] = self.sensorMatSize - 1
        yIdx[yIdx >= self.sensorMatSize] = self.sensorMatSize - 1
        sensorMat = np.zeros((self.sensorMatSize, self.sensorMatSize), dtype=np.int32)
        sensorMat[xIdx, yIdx] = 1
        
        return sensorMat
                
    def getOrientationHist(self, partConfig):

        hist, _ = computeOrienHist(partConfig)
        hist = hist / self.N
        return alginOrienHist(self.refHist, hist)
    
     


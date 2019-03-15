#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:53:48 2019

@author: yangyutu123
"""

import numpy as np
import Model_QuadrupoleBD as BDModel
import random

class BDQuadModelEnv:
    def __init__(self, config, randomSeed):
        self.config = config
        self.model = BDModel.Model_QuadrupoleBD('trajTest', 0, randomSeed)
        self.nbActions = 4
        self.stateDim = 600

        self.initConfigFileTag = self.config['BDModelConfigFile']
        self.initConfigFileRange = self.config['BDModelConfigFileRange']
        self.episodeCount = 0
    def reset(self):
        self.maxPsi6 = 0.0
        fileCount = self.episodeCount % self.initConfigFileRange
        initFile = self.initConfigFileTag + str(fileCount) + '.txt'
        self.model.setInitialConfigFile(initFile)
        self.model.createInitialState()
        partConfig = model.getParticleConfig()
        
        self.episodeCount += 1
        
        return partConfig
        
    def step(self, action):
        self.model.run(action)
        partConfig = model.getParticleConfig()
        info = model.getInfo()
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
        
        

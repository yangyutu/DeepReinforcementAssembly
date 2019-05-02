#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:20:49 2019

@author: yangyutu123
"""

import cppimport
import numpy as np
import timeit
BDModel = cppimport.imp("Model_QuadrupoleBD")


#model = BDModel.Model_QuadrupoleBD('good',1)


model = BDModel.Model_QuadrupoleBD('trajTestN60_', 2, 1, 60)

model.setInitialConfigFile('RandomConfigN300_0.txt')

model.createInitialState()

partConfig = model.getParticleConfig()
info = model.getInfo()
print(info)
start_time = timeit.default_timer()
for i in range(100):
    print(i)
    model.run(3)
   # partConfig = model.getParticleConfig()
    info = model.getInfo()
    #print(partConfig)
    print(info)


elapsed = timeit.default_timer() - start_time

print(elapsed)
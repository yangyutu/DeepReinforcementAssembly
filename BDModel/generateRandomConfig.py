#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 20:45:29 2019

@author: yangyutu123
"""

import math
import numpy as np
from collections import deque
import matplotlib.pylab as plt

import numpy as np
scale = 2
vec1 = np.array([2.0, 0.0])*2
vec2 = np.array([2.0 * math.cos(np.pi / 3.0), 2.0 * math.sin(np.pi / 3.0)])*2

q = deque()
q.append(np.array([0.0, 0.0]))

N = 1000
config = []

# put in first line
for i in range(50):
    p = q[-1]
    p1 = p + vec1 + (np.random.random(2) - np.array([0.5, 0.5]))
    q.append(p1)

for p in q:
    config.append(p)

for i in range(N):
    p = q[0]
    q.popleft()
    p2 = p + vec2 + (np.random.random(2) - np.array([0.5, 0.5]))
    q.append(p2)
    config.append(p2)

config = np.array(config)

plt.close('all')
plt.figure(1)

plt.scatter(config[:, 0], config[:, 1])

config[:, 0] = config[:, 0] - np.mean(config[:, 0])
config[:, 1] = config[:, 1] - np.mean(config[:, 1])

dist = np.sqrt(config[:, 0] ** 2 + config[:, 1] ** 2)

distSortIdx = np.argsort(dist)

N = 300

configOut = config[distSortIdx[0:N], :]

plt.figure(2)
plt.scatter(configOut[:, 0], configOut[:, 1])

output = np.array([list(np.arange(N)), list(configOut[:,0]), list(configOut[:,1]), list(np.zeros(N))]).T

np.savetxt('RandomConfigN300_0.txt', output)
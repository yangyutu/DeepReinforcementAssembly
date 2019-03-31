import numpy as np


with np.load('opfilep300.npz') as data:
    opSampleAll = data['a']
    xyzSampleAll = data['b']

psi6 = opSampleAll[:, 1]


config = []





idx = np.where((psi6 > 0.7) & (psi6 < 0.8))
idxSelect = np.random.choice(idx[0], 10)
config = config + xyzSampleAll[idxSelect].tolist()

idx = np.where((psi6 > 0.5) & (psi6 < 0.7))
idxSelect = np.random.choice(idx[0], 20)
config = config + xyzSampleAll[idxSelect].tolist()

idx = np.where(psi6 < 0.5)
idxSelect = np.random.choice(idx[0], 70)
config = config + xyzSampleAll[idxSelect].tolist()

config = np.array(config)

for i, part in enumerate(config):
    partAug = np.hstack((np.reshape(np.arange(300),(300,1)), part, np.zeros((300,1))))
    np.savetxt('iniConfig'+str(i)+'.txt', partAug)

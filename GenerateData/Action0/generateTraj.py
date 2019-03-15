

import BDModel.Model_QuadrupoleBD as BDModel


action = 0


model = BDModel.Model_QuadrupoleBD('trajAction' + str(action), 1, 1)

model.setInitialConfigFile('../Startmeshgrid1.txt')



partConfig = model.getParticleConfig()
info = model.getInfo()
ntraj = 100


for i in range(ntraj):
    model.createInitialState()
    for i in range(500):
        model.run(action)

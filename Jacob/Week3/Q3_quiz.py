import numpy as np

D_d = 1e-3 # m, diameter of glass beads
nu_c = 1e-6 # m^2/s, kinematic viscosity of carrier fluid

u = 0 # m/s, velocity of the fluid
v = 200 # m/s, velocity of the particle

Re_r = D_d * np.abs(u-v) / nu_c

stop = True

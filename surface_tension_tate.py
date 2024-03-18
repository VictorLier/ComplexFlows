import numpy as np

g = 9.81
D = 1.6e-3

m_water = 2e-3/60 # total mass divided by number of drops
m_alcohol = 1e-3/70 # total mass divided by number of drops

gamma_water = m_water*g/(D*np.pi)
gamma_alcohol = m_alcohol*g/(D*np.pi)

gamma_water_known = 75.6e-3

gamma_alcohol_ratio = gamma_water_known * (m_alcohol/m_water)

stop = True


import numpy as np

D_d = 1e-3 # m, diameter of glass beads
rho_c = 1.2 # kg/m^3, density of carrier fluid
rho_d = 2500 # kg/m^3, density of particles
nu_c = 1.51e-5 # m^2/s, kinematic viscosity of carrier fluid
g = 9.81 # m/s^2, acceleration due to gravity


# Archimedes number
Ar = (rho_d/rho_c-1) * g * D_d**3 / nu_c**2

if Ar < 3*10**5:
    v_T = nu_c/D_d*(np.sqrt(22+np.sqrt(4.89*Ar))-np.sqrt(22))**2
elif 3*10**5 <= Ar and Ar < 3*10**10:
    v_T = 1.74*nu_c/D_d*np.sqrt(Ar)
else:
    v_T = None
    print('The Archimedes number is too large for the empirical formula to be valid')

stop = True

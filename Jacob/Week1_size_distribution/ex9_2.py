import numpy as np

error = 0.05 # coincedence error

C = 2 # Mass loading ratio (dispersed phase mass concentration)

rho_d = 2500 # kg/m^3, density of particles
rho_c = 1.2 # kg/m^3, density of carrier fluid

D_d1 = 15e-6 # m, diameter of particles
D_d2 = 60e-6 # m, diameter of particles

# Masses of particles
m_dot_d1 = np.pi/6 * rho_d * D_d1**3
m_dot_d2 = np.pi/6 * rho_d * D_d2**3

# Number densities of particles
n_D1 = C * rho_c / (m_dot_d1)
n_D2 = C * rho_c / (m_dot_d2)

# Probe volumes (eq 9.37)
V_m_D1 = 0.1/n_D1
V_m_D2 = 0.1/n_D2

L_d1 = V_m_D1**(1/3)
L_d2 = V_m_D2**(1/3)

stop = True


import numpy as np

D = 100e-6 # m
l = 0.02e-4 # m/s
Te = D**2/l

D_pipe = 5e-2 # m
A_pipe = np.pi/4*D_pipe**2
m_dot_c = 0.1 # kg/s
rho_c = 1.2 # kg/m^3

u = m_dot_c/(rho_c*A_pipe)

St_mass = Te*u/D_pipe

v = 30 # m/s
m_dot_d = 0.01 # m/s
z = m_dot_d/m_dot_c

C = z*u/v

PI_mass = C/St_mass

cp_c = 1005 # J/(kg*K)
T_c = 293.15 # K




stop = True
import numpy as np

D_pipe = 200e-3 # m
D_probe = 10e-3 # m
t_sample = 60 # s
D_d = 200e-6 # m
rho_d = 2500 # kg/m^3
rho_c = 1.18 # kg/m^3
nu_c = 18.4e-6 # Ns/m^2
mu_c =  nu_c/ rho_c # m^2/s

u = 20 # m/s
v = u

M_sample = 60e-3 # kg

M_dot_d = M_sample / t_sample

A_probe = np.pi * D_probe**2 / 4
rho_bulk_d = M_dot_d / (A_probe * u)

n = 7
u_0 = u*(n+1)*(2*n+1)/(2*n**2)
y = 50e-3
u_50 = (y/(D_pipe/2))**(1/7)*u_0

B = rho_d*D_d**2*u_0/(18*mu_c*D_probe)
B_50 = rho_d*D_d**2*u_50/(18*mu_c*D_probe)

# At the center of the pipe
u_ratio_center = u/u_0
# From figure 9.26
rho_bulk_ratio_center = 1.05
rho_bulk_center = rho_bulk_d/rho_bulk_ratio_center # Measured value

# At 50 mm from the center
u_ratio_50 = u/u_50
# From figure 9.26
rho_bulk_ratio_50 = 1.1
rho_bulk_50 = rho_bulk_d/rho_bulk_ratio_50 # Measured value

stop = True
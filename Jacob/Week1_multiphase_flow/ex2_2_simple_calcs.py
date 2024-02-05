import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

rho_d_bulk = 0.5 # kg/m^3
D = 100e-6 # m
rho_d = 1000 # kg/m^3

alpha_d = rho_d_bulk/rho_d
alpha_c = 1 - alpha_d

rho_c = 1.2 # kg/m^3
rho_c_bulk = alpha_c*rho_c

C = rho_d_bulk/rho_c_bulk

Vol_d = np.pi/6*D**3
n = alpha_d/Vol_d

stop = True
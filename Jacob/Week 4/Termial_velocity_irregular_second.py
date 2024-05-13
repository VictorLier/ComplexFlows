from scipy.optimize import fsolve, minimize_scalar
import numpy as np

# Define constants
g = 9.81  # m/s^2
rho_d = 1400.0  # kg/m^3
rho_c = 1.20  # kg/m^3
nu_c = 1.50 * 10**(-5)  # m^2/s
mu_c = nu_c * rho_c  # kg/(m*s)
L = 1.0 / 1000  # m (converted from mm to m)
Ap = L**2  # m^2
Vd = L**3  # m^3
D_V = (6.0 * Vd / np.pi)**(1/3)  # m
D_PA = (4.0 * Ap / np.pi)**(1/2)  # m
P_P = 4.0 * L  # m
# Calculate CD
c = np.pi * D_PA / P_P
tau_v = rho_d * D_V**2 / (18 * mu_c)

# Function to find the root of
def find_vT(_vT):
    # Calculate Re_r
    Re_r = D_V * _vT / nu_c
    
    f = D_PA / D_V * (1 + 0.15 / (np.sqrt(c) * (Re_r * D_PA / D_V)**0.687)) + 0.42*(D_PA/D_V)** 2* Re_r / (np.sqrt(c) * (1 + 4.25 * 10**4 * (Re_r * D_PA / D_V)**(-1.16)))
    
    # Calculate vT
    vT = g*tau_v/f
    
    return  np.abs(vT - _vT)

# Initial guess for _vT
_vT_guess = 1  # m/s

# Use fsolve to find the root
_vT_solution = minimize_scalar(find_vT, _vT_guess, [0, 25])

print("Converged _vT:", _vT_solution)
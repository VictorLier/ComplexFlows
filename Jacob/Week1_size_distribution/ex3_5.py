import numpy as np
import sympy as sym 

D, lambda_ = sym.symbols('D lambda')

f_n = 0.01
D_max = 100

# Cumulative distribution function
F = sym.integrate(f_n, (lambda_, 0, D))

# Number mean and standard deviation
mu_N = sym.integrate(D * f_n, (D, 0, D_max))
sigma_N = sym.sqrt(sym.integrate((D - mu_N)**2 * f_n, (D, 0, D_max)))

# Sauter mean diameter
D_32 = sym.integrate(D**3 * f_n, (D, 0, D_max)) / sym.integrate(D**2 * f_n, (D, 0, D_max))

# Number median diameter
D_nM = float(sym.solve(F - 0.5, D)[0])

# Mass frequency distribution
f_M = f_n * D**3/ sym.integrate(D**3 * f_n, (D, 0, D_max))

# Mass mode diameter
# D_M = float(sym.solve(sym.diff(f_M, D), D)[0])
# Numerical maximum of f_M
d_m_mode = np.linspace(0, D_max, 1000)
f = np.zeros(len(d_m_mode))
for i in range(len(d_m_mode)):
    f[i] = f_M.subs(D, d_m_mode[i])
D_m_mode = d_m_mode[np.argmax(f)]

# Mass mean diameter and standard deviation
mu_M = sym.integrate(D * f_M, (D, 0, D_max))
sigma_M = sym.sqrt(sym.integrate((D - mu_M)**2 * f_M, (D, 0, D_max)))

# Mass cumulative distribution function
F_M = sym.integrate(f_M, (D, 0, D))

# Mass median diameter, get positive and real solution
D_mM = sym.solve(F_M - 0.5, D)
for i in range(len(D_mM)):
    if D_mM[i].is_real and D_mM[i] > 0:
        D_mM = float(D_mM[i])
        break

stop = True

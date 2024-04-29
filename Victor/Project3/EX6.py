import sympy as sp

# Define the variables
L, mu_g, mu_l, H, gamma, theta, rho_l, rho_g, t = sp.symbols('L mu_g mu_l H gamma theta rho_l rho_g t')

# Define the function
l = sp.Function('l')
eq1_lhs = ( -mu_g * L - l(t) * mu_l ) * sp.Derivative(l(t),t) / H + gamma * sp.cos(theta)
eq1_rhs = (rho_l * l(t) + rho_g * L) * sp.Derivative(l(t),t, t)

eq1 = sp.Eq(eq1_lhs, eq1_rhs)

ics = {l(0): 0, sp.Derivative(l(t),t).subs(t,0): 0}
sol = sp.dsolve(eq1, ics=ics)
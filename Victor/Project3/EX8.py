import sympy as sp

# Define the variables
L, B, A, t = sp.symbols('L B A t')

# Define the function
l = sp.Function('l')
eq1 = sp.Eq(sp.Derivative(L * sp.Derivative(l(t),t), t) + B * L * sp.Derivative(l(t),t), A**2)
ics = {l(0): 0, sp.Derivative(l(t),t).subs(t,0): 0}

sol = sp.dsolve(eq1, ics=ics)

l_func = sp.Lambda((t, A, B, L), sol.rhs)

stop = True
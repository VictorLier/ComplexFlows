import sympy as sp

# Define the symbols
alpha, L, Dp = sp.symbols('alpha L Dp')

# Define the equation
eq = sp.Eq(alpha*L**3, sp.pi/6*Dp**3)



# Solve the equation
L = sp.solve(eq, L)[2]
L_DP = L/Dp

One = L_DP.subs({alpha: 5e-4}).evalf()
Two = L_DP.subs({alpha: 0.1}).evalf()


print("One way coupiling is valid for L/Dp < ", One,"two way coupling is valid for L/Dp < ", Two, "Four way coupling is valid for L/Dp >", Two)
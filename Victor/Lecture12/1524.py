import numpy as np

def oblique_shock_velocity(delta, sigma, M1):
    '''
    Calculates the velocity after an oblique shock
    delta: float: deflection angle in degrees
    sigma: float: wave angle in degrees
    '''
    delta = np.radians(delta)
    sigma = np.radians(sigma)

    v = np.cos(sigma)
    u_1 = np.sin(sigma)
    u_2 = v * np.tan(sigma-delta)
    M2 = np.sqrt(v**2 + u_2**2)
    return M2


# Values from figure 15.22
sigma = 20
alpha = 53
M1 = 2
M2 = oblique_shock_velocity(alpha, sigma, M1)
print(M2)
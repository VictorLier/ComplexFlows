import numpy as np
# Define f and e
f = 0
e = 0

# Defining particle 1
r_i = 1
m_i = 1
x_i = np.array([0, 0, 0])
v_i = np.array([2, 0, 0])
omega_i = np.array([0, 0, -1])

# Defining particle 2
r_j = 2
m_j = 2
x_j = np.array([1.5, 0, 0])
v_j = np.array([-1, 0, 0])
omega_j = np.array([0, 0, 1])


distance = np.sqrt(((x_j - x_i)**2).sum())
if not distance <= (r_i + r_j):
    raise ValueError("Particles are not in contact")

n = (x_j - x_i) / distance # unit vector between the i-th and j-th particle at the moment of contact
G0 = v_i - v_j # relative velocity at the moment of contact


G0_c = G0 + np.cross(r_i*omega_i, n) + np.cross(r_j*omega_j, n) # relative velocity at the moment of contact in the contact frame
G0_ct = G0_c - np.dot(G0_c, n) * n # relative velocity at the moment of contact in the tangential frame

t = G0_ct / np.linalg.norm(G0_ct) # unit vector in the tangential frame
if f == 0 or np.dot(n, G0)/np.linalg.norm(G0_ct) < (2/7)*1/(f * (1 + e)): 
    print("Continuous sliding")
    v_i_1 = v_i - (n + f * t) * np.dot(n, G0) * (1 + e) * m_j / (m_i + m_j)
    v_j_1 = v_j + (n + f * t) * np.dot(n, G0) * (1 + e) * m_i / (m_i + m_j)

    omega_i_1 = omega_i - 5/(2*r_i) * np.dot(n, G0) * np.cross(n, t) * f * (1 + e) * m_j / (m_i + m_j)
    omega_j_1 = omega_j - 5/(2*r_j) * np.dot(n, G0) * np.cross(n, t) * f * (1 + e) * m_i / (m_i + m_j)
else:
    print("None continuous sliding")
    v_i_1 = v_i - ((1 + e) * np.dot(n, G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_j / (m_i + m_j)
    v_j_1 = v_j + ((1 + e) * np.dot(n, G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_i / (m_i + m_j)

    omega_i_1 = omega_i - 5/(7*r_i) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_j / (m_i + m_j)
    omega_j_1 = omega_j - 5/(7*r_j) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_i / (m_i + m_j)

stop = True
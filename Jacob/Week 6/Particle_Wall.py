import numpy as np
import matplotlib.pyplot as plt
import time
import os
from typing import Tuple, List

def get_v_1_and_o_1(radius, v, o, n) -> Tuple[np.ndarray, np.ndarray]:
    f = 0.2 # friction coefficient
    e = 1 # restitution coefficient
    # Calculate the new velocity and angular velocity of the particle after collision with the wall
    G0 = v # relative velocity at the moment of contact, v2= 0 because the wall is stationary
    G0_c = G0 + np.cross(radius*o, n)  # relative velocity at the moment of contact in the contact frame
    G0_ct = G0_c - np.dot(G0_c, n) * n # relative velocity at the moment of contact in the tangential frame

    t = G0_ct / np.linalg.norm(G0_ct) # unit vector in the tangential frame

    if f == 0 or np.dot(n, G0)/np.linalg.norm(G0_ct) < (2/7)*1/(f * (1 + e)):
        # Continuous sliding 
        v_1 = v - (n + f * t) * np.dot(n, G0) * (1 + e) 
        o_1 = o - 5/(2*radius) * np.dot(n, G0) * np.cross(n, t) * f * (1 + e)
    else:
        # Non-continuous sliding
        v_1 = v - ((1 + e) * np.dot(n, G0) * n + 2/7 * np.linalg.norm(G0_ct) * t)
        o_1 = o - 5/(7*radius) * np.linalg.norm(G0_ct) * np.cross(n, t)

    return v_1, o_1            

box_limits = np.array([0, 10])
x = np.array([5, 1, 3])
v = np.array([-1, -1, 0])
o = np.array([0, 0, -1])
radius = 1

west_collision = x[0] - radius <= box_limits[0]
east_collision = x[0] + radius >= box_limits[1]
south_collision = x[1] - radius <= box_limits[0]
north_collision = x[1] + radius >= box_limits[1]

if west_collision:
    # Unit vector in the direction of the wall
    n = np.array([-1, 0, 0])

if east_collision:
    # unit vector in the direction of the wall
    n = np.array([1, 0, 0])
    
if south_collision:
    # Unit vector in the direction of the wall
    n = np.array([0, - 1, 0])

if north_collision:
    # Unit vector in the direction of the wall
    n = np.array([0, 1, 0])

v_1, o_1 = get_v_1_and_o_1(radius, v, o, n)
stop = True
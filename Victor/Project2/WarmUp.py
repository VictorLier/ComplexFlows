import numpy as np
from typing import List
import sympy as sp
import time

#t, r1, r2,  vx1, x1, x2, vx2, y1, y2, vy1, vy2 = sp.symbols('t r1 r2 vx1 x1 x2 vx2 y1 y2 vy1 vy2')

#timetest = sp.solve(sp.sqrt((-t*vx1 + t*vx2 - x1 + x2)**2 + (-t*vy1 + t*vy2 - y1 + y2)**2) - r1 - r2,t)



def time_collision(r_1, r_2, x_1: list[float, float], x_2: list[float, float], v_1: list[float, float], v_2: list[float, float]) -> float | None:
    '''
    Computes the time of collision between two particles.
    Returns time, None if no collision
    Inputs:
    radius: r_1, r_2,
    Vector of location: x_1, x_2
    Velocity vector: v_1, v_2
    '''
    closing_speed_0 = np.linalg.norm(v_1 - v_2)
    
    if closing_speed_0 <= 0:
        t = None
        return t

    x1 = x_1[0]
    y1 = x_1[1]
    x2 = x_2[0]
    y2 = x_2[1]

    vx1 = v_1[0]
    vy1 = v_1[1]
    vx2 = v_2[0]
    vy2 = v_2[1]

    r1 = r_1
    r2 = r_2

    st = time.perf_counter()
    t = (-vx1*x1 + vx1*x2 + vx2*x1 - vx2*x2 - vy1*y1 + vy1*y2 + vy2*y1 - vy2*y2 - np.sqrt(-vx1**2*y1**2 + 2*vx1**2*y1*y2 - vx1**2*y2**2 + vx1**2*(r1 + r2)**2 + 2*vx1*vx2*y1**2 - 4*vx1*vx2*y1*y2 + 2*vx1*vx2*y2**2 - 2*vx1*vx2*(r1 + r2)**2 + 2*vx1*vy1*x1*y1 - 2*vx1*vy1*x1*y2 - 2*vx1*vy1*x2*y1 + 2*vx1*vy1*x2*y2 - 2*vx1*vy2*x1*y1 + 2*vx1*vy2*x1*y2 + 2*vx1*vy2*x2*y1 - 2*vx1*vy2*x2*y2 - vx2**2*y1**2 + 2*vx2**2*y1*y2 - vx2**2*y2**2 + vx2**2*(r1 + r2)**2 - 2*vx2*vy1*x1*y1 + 2*vx2*vy1*x1*y2 + 2*vx2*vy1*x2*y1 - 2*vx2*vy1*x2*y2 + 2*vx2*vy2*x1*y1 - 2*vx2*vy2*x1*y2 - 2*vx2*vy2*x2*y1 + 2*vx2*vy2*x2*y2 - vy1**2*x1**2 + 2*vy1**2*x1*x2 - vy1**2*x2**2 + vy1**2*(r1 + r2)**2 + 2*vy1*vy2*x1**2 - 4*vy1*vy2*x1*x2 + 2*vy1*vy2*x2**2 - 2*vy1*vy2*(r1 + r2)**2 - vy2**2*x1**2 + 2*vy2**2*x1*x2 - vy2**2*x2**2 + vy2**2*(r1 + r2)**2))/(vx1**2 - 2*vx1*vx2 + vx2**2 + vy1**2 - 2*vy1*vy2 + vy2**2)
    et = time.perf_counter()


    print(et-st)

    if t < 0:
        t = None
    return t




if __name__ == "__main__":

    r_1 = 1
    r_2 = 1
    x_1 = np.array([0,0])
    x_2 = np.array([10,4])
    v_1 = np.array([-1,2])
    v_2 = np.array([1,1])

    t = time_collision(r_1,r_2,x_1, x_2, v_1, v_2)


    print(t)
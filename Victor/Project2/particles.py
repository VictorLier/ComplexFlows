# -*- coding: utf-8 -*-
# Knud Erik Meyer, DTU Construct, 27/2-2023

import numpy as np
from numpy import sqrt, cos, sin, array, zeros
import pyglet

#parameters
window_size = np.array((1920, 1080))
nparticles = 100
radius = 20
max_velocity = 200
max_omega = 0.5
color1 = (20, 20, 250) # blue
color2 = (200, 200, 250) # light blue

# initialize global variables for particles
position = zeros((nparticles, 2))
velocity = zeros((nparticles, 2))
angle = zeros(nparticles)
omega = zeros(nparticles)
circles = []

# make window and batch
window = pyglet.window.Window(window_size[0], window_size[1])
batch = pyglet.graphics.Batch()

# functions for particle simulation

def make_particles(position, velocity, omega, circles):
    # make particles in a grid at least one diameter from walls  
    xstart =  np.ones(2) * radius * 2
    xlength = window_size - 2 * xstart
    n = np.ceil(np.sqrt(nparticles))
    m = np.ceil(nparticles / n)
    xg, yg = np.meshgrid(np.arange(n)/(n-1), np.arange(m)/(m-1)) 
    # make random velocities and rotation
    velangle = np.random.rand(nparticles) * 2 * np.pi
    velocity += np.random.rand(nparticles).reshape(-1,1) * max_velocity \
                * array([cos(velangle), sin(velangle)]).T 
    omega += (np.random.rand(nparticles) - 0.5) * 2 * max_omega
    # prepare particles for simulation   
    for i in range(nparticles):
        position[i,0] = xg.flatten()[i] * xlength[0] + xstart[0]
        position[i,1] = yg.flatten()[i] * xlength[1] + xstart[1]
        # generate circle for particle
        circles.append(pyglet.shapes.Circle(position[i,0], position[i,1], 
                                            radius, color=color1, batch=batch))
        # generate spot on particle to track rotation
        x, y = position[i,:] + 0.5 * radius * array([cos(angle[i]), sin(angle[i])])
        circles.append(pyglet.shapes.Circle(x, y, 0.25*radius, 
                                            color=color2, batch=batch))
def move_particles(dt, position, angle, circles):
    position += velocity * dt
    angle += omega * dt
    # update circle positions
    for i in range(nparticles):
        circles[i*2].position = position[i]
        circles[i*2+1].position = position[i] + 0.5 * radius * array([cos(angle[i]), 
                                                               sin(angle[i])])
        
def wall_collision(velocity):
    # handle particle collition with the walls
    for i in range(nparticles):
        if position[i,0] < radius:
            velocity[i,0] = abs(velocity[i,0])
        if position[i,0] > window_size[0] - radius :
            velocity[i,0] = -abs(velocity[i,0])
        if position[i,1] < radius:
            velocity[i,1] = abs(velocity[i,1])
        if position[i,1] > window_size[1] - radius :
            velocity[i,1] = -abs(velocity[i,1])

def particle_collision(velocity):
    # handle collision between particles using simple loops
    for i in range(nparticles-1):
        for j in range(i+1, nparticles):
            distance = sqrt(((position[j] - position[i])**2).sum())
            if  distance < 2 * radius:
                # collision! - apply textbook eq. 5.14
                n = (position[j] - position[i]) / distance
                G0 = velocity[i] - velocity[j]
                dvel = n * np.dot(n, G0)
                velocity[i] -= dvel
                velocity[j] += dvel
            
# modify pyglet draw command to draw our particles
@window.event
def on_draw():
    window.clear()
    batch.draw()

# This is want we do in each time step
def update(dt):
    move_particles(dt, position, angle, circles)
    wall_collision(velocity)
    particle_collision(velocity)

# run the following if this is the main script
if __name__ == "__main__":

    # Update the game 120 times per second
    pyglet.clock.schedule_interval(update, 1 / 120.0)
    
    # Create particles with random position and velocity
    make_particles(position, velocity, omega, circles)

    # Tell pyglet to do its thing
    pyglet.app.run()

    del window
    del batch
 

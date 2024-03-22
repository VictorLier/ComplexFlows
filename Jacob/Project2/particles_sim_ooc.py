import numpy as np
import matplotlib.pyplot as plt
import pyglet

class particle():
    def __init__(self, radius = 20, position = np.array((0, 0)), velocity = np.array((0, 0)), angle = 0, omega = 0):
        
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.color1 = (20, 20, 250)
        self.color2 = (200, 200, 250)
        self.angle = angle
        self.omega = omega
        self.circle = None
        self.circle2 = None

    def move(self, dt):
        self.position += self.velocity * dt
        self.angle += self.omega * dt
        self.angle = self.angle % (2 * np.pi)

    def update_pyglet(self):
        self.circle.position = self.position[0], self.position[1]
        self.circle2.position = self.position[0] + 0.5 * self.radius * np.cos(self.angle), self.position[1] + 0.5 * self.radius * np.sin(self.angle)

    def wall_collision(self, window_size):
        # handle particle collition with the walls
        if self.position[0] < self.radius:
            self.velocity[0] = abs(self.velocity[0])
        if self.position[0] > window_size[0] - self.radius :
            self.velocity[0] = -abs(self.velocity[0])
        if self.position[1] < self.radius:
            self.velocity[1] = abs(self.velocity[1])
        if self.position[1] > window_size[1] - self.radius :
            self.velocity[1] = -abs(self.velocity[1])

class sim_particles():
    def __init__(self, window_size = np.array((800, 800)), nparticles = 20, radius = 20, 
                 max_velocity = 200, max_omega = 0.5, color1 = (20, 20, 250), 
                 color2 = (200, 200, 250)):
        
        self.window_size = window_size
        self.nparticles = nparticles
        self.radius = radius
        self.max_velocity = max_velocity
        self.max_omega = max_omega
        self.color1 = color1
        self.color2 = color2
        self.particles: list[particle] = []
        self.window_size = window_size
        self.make_particles()
        
        
    def make_particles(self):
        # make particles in a grid at least one diameter from walls  
        self.xstart =  np.ones(2) * self.radius * 2
        self.xlength = self.window_size - 2 * self.xstart
        n = np.ceil(np.sqrt(self.nparticles))
        m = np.ceil(self.nparticles / n)
        self.xg, self.yg = np.meshgrid(np.arange(n)/(n-1), np.arange(m)/(m-1)) 
        xg_f = self.xg.flatten()
        yg_f = self.yg.flatten()
        # Make particles in a grid at least one diameter from walls with the use of the particle class
        position = np.zeros((self.nparticles, 2))
        for i in range(self.nparticles):
            position[i,0] = self.xg.flatten()[i] * self.xlength[0] + self.xstart[0]
            position[i,1] = self.yg.flatten()[i] * self.xlength[1] + self.xstart[1]
        velocity = np.zeros((self.nparticles, 2))
        velangle = np.random.rand(self.nparticles) * 2 * np.pi
        velocity += np.random.rand(self.nparticles).reshape(-1,1) * self.max_velocity \
                    * np.array([np.cos(velangle), np.sin(velangle)]).T
        omega = (np.random.rand(self.nparticles) - 0.5) * 2 * self.max_omega
        for i in range(self.nparticles):
            self.particles.append(particle(self.radius, position[i], velocity[i], 0, omega[i]))

    def make_particles_pyglet(self, batch):
        for i in range(self.nparticles):
            self.particles[i].circle = pyglet.shapes.Circle(self.particles[i].position[0], 
                                                            self.particles[i].position[1], 
                                                            self.radius, color = self.color1, batch = batch)
            self.particles[i].circle2 = pyglet.shapes.Circle(self.particles[i].position[0] + 0.5 * self.radius * np.cos(self.particles[i].angle), 
                                                            self.particles[i].position[1] + 0.5 * self.radius * np.sin(self.particles[i].angle), 
                                                            0.25 * self.radius, color = self.color2, batch = batch)
    def move_particles(self, dt):
        for i in range(self.nparticles):
            self.particles[i].move(dt)

    def update_pyglet(self):
        for i in range(self.nparticles):
            self.particles[i].update_pyglet()
    
    def wall_collision(self):
        # handle particle collition with the walls
        for i in range(self.nparticles):
            self.particles[i].wall_collision(self.window_size)

    def particle_collision(self):
        for i in range(self.nparticles-1):
            for j in range(i+1, self.nparticles):
                # distance = np.sqrt(((self.position[j] - self.position[i])**2).sum())
                # if  distance < 2 * self.radius:
                #     # collision! - apply textbook eq. 5.14
                #     n = (self.position[j] - self.position[i]) / distance
                #     G0 = self.velocity[i] - self.velocity[j]
                #     dvel = n * np.dot(n, G0)
                #     self.velocity[i] -= dvel
                #     self.velocity[j] += dvel
                distance = np.sqrt(((self.particles[j].position - self.particles[i].position)**2).sum())
                if  distance < 2 * self.radius:
                    # collision! - apply textbook eq. 5.14
                    n = (self.particles[j].position - self.particles[i].position) / distance
                    G0 = self.particles[i].velocity - self.particles[j].velocity
                    dvel = n * np.dot(n, G0)
                    self.particles[i].velocity -= dvel
                    self.particles[j].velocity += dvel

    def update(self, dt):
        self.move_particles(dt)
        self.wall_collision()
        self.particle_collision()

    def update_to_pyglet(self, dt):
        self.move_particles(dt)
        self.update_pyglet()
        self.wall_collision()
        self.particle_collision()



if __name__ == '__main__':
    sim = sim_particles()

    if True:
        window = pyglet.window.Window(sim.window_size[0], sim.window_size[1])
        batch = pyglet.graphics.Batch()

        @window.event
        def on_draw():
            window.clear()
            batch.draw()

        pyglet.clock.schedule_interval(sim.update_to_pyglet, 1/120.0)

        sim.make_particles_pyglet(batch)
        
        pyglet.app.run()

        del window
        del batch
            
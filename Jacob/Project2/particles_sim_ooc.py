import numpy as np
import matplotlib.pyplot as plt
import pyglet

# Make seed for reproducibility
np.random.seed(1917)

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
    def __init__(self, window_size = np.array((800, 800)), nparticles = 20, radius = 20):
        
        self.window_size = window_size
        self.nparticles = nparticles
        self.radius = radius
        self.max_velocity = 200
        self.max_omega = 0.5
        self.color1 = (20, 20, 250)
        self.color2 = (200, 200, 250)
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
                distance = np.sqrt(((self.particles[j].position - self.particles[i].position)**2).sum())
                if  distance < 2 * self.radius:
                    # collision! - apply textbook eq. 5.14
                    n = (self.particles[j].position - self.particles[i].position) / distance
                    G0 = self.particles[i].velocity - self.particles[j].velocity
                    dvel = n * np.dot(n, G0)
                    self.particles[i].velocity -= dvel
                    self.particles[j].velocity += dvel

    def time_before_collision(self, particle_a: particle, particle_b: particle):
        r1 = particle_a.radius
        r2 = particle_b.radius
        
        x1 = particle_a.position
        x2 = particle_b.position
        v1 = particle_a.velocity
        v2 = particle_b.velocity
        
        delta_x = x2 - x1
        delta_v = v2 - v1
        closing_speed_0 = np.linalg.norm(delta_v)
        
        if closing_speed_0 <= 0:
            return np.NaN
        
        a = np.dot(delta_v, delta_v)
        b = 2 * np.dot(delta_v, delta_x)
        c = np.dot(delta_x, delta_x) - (r1 + r2) ** 2
        
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            return np.NaN
        
        t = (-b - np.sqrt(discriminant)) / (2 * a)
        
        if t < 0 or np.isnan(t):
            return np.NaN
        return t
    
    def time_before_collision_wall(self, particle: particle):
        r = particle.radius
        x = particle.position[0]
        y = particle.position[1]
        vx = particle.velocity[0]
        vy = particle.velocity[1]
        
        if vx > 0:
            t_x = (self.window_size[0] - r - x) / vx
        elif vx < 0:
            t_x = (r - x) / vx
        else:
            t_x = np.NaN
        
        if vy > 0:
            t_y = (self.window_size[1] - r - y) / vy
        elif vy < 0:
            t_y = (r - y) / vy
        else:
            t_y = np.NaN
        
        if not np.isnan(t_y):
            if not np.isnan(t_x):
                return min(t_x, t_y)
        return np.NaN
    
    def init_time_before_collision_matrix(self):
        self.tau_particle_matrix = np.full((self.nparticles, self.nparticles), np.NaN)
        for i in range(self.nparticles-1):
            for j in range(i+1, self.nparticles):
                tau = self.time_before_collision(self.particles[i], self.particles[j])
                self.tau_particle_matrix[j,i] = tau
        print(np.nanmin(self.tau_particle_matrix), np.nanargmin(self.tau_particle_matrix))
        
        self.tau_wall_matrix = np.full(self.nparticles, np.NaN)
        for i in range(self.nparticles):
            self.tau_wall_matrix[i] = self.time_before_collision_wall(self.particles[i])

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
    _window_size = np.array((800, 800))
    _nparticles = 10
    _radius = 20
    sim = sim_particles(_window_size, _nparticles, _radius)

    if False:
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

    if True:
        sim.init_time_before_collision_matrix()
        stop = True
            
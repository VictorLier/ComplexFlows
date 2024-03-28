import numpy as np
import matplotlib.pyplot as plt
import pyglet



class particle():
    def __init__(self, e = 0.5, f = 0.5, rho = 1, rho_c = 1, radius = 20, position = np.array((0, 0)), velocity = np.array((0, 0)), angle = 0, omega = 0):
        self.e = e
        self.f = f
        self.rho = rho
        self.rho_c = rho_c
        self.position = position
        self.velocity = velocity
        self.radius = radius
        self.mass = 4/3 * np.pi * self.radius ** 3 * self.rho
        self.I = 2/5 * self.mass * self.radius ** 2
        self.color1 = (20, 20, 250)
        self.color2 = (200, 200, 250)
        self.angle = angle
        self.omega = omega
        self.circle = None
        self.circle2 = None

        self.t = None

        self.cs = self.f == 0 # use continuous sliding model if f = 0

        self.in_wall_collision = [False, False, False, False]

        self.wall_res_fric = False # If True enable friction and restitution on the walls
        self.drag = False # If True enable drag force

    @property
    def C_D(self):
        return 0.1 # drag coefficient
    
    @property
    def F_D(self):
        F_D_x = -0.5 * self.C_D * self.rho_c * np.pi * self.radius ** 2 * np.linalg.norm(self.velocity) * self.velocity[0]
        F_D_y = -0.5 * self.C_D * self.rho_c * np.pi * self.radius ** 2 * np.linalg.norm(self.velocity) * self.velocity[1]
        return np.array([F_D_x, F_D_y])
    @property
    def a_D(self):
        return self.F_D / self.mass

    def move(self, dt):
        self.position += self.velocity * dt # update position
        self.angle += self.omega * dt # update angle
        self.angle = self.angle % (2 * np.pi) # keep angle between 0 and 2*pi
        if self.drag:
            self.velocity += self.a_D * dt # drag force

    def update_pyglet(self):
        self.circle.position = self.position[0], self.position[1]
        self.circle2.position = self.position[0] + 0.5 * self.radius * np.cos(self.angle), self.position[1] + 0.5 * self.radius * np.sin(self.angle)

    def wall_collision(self, window_size):
        # handle particle collition with the walls
        west_collision = self.position[0] < self.radius
        east_collision = self.position[0] > window_size[0] - self.radius
        south_collision = self.position[1] < self.radius
        north_collision = self.position[1] > window_size[1] - self.radius

        if self.wall_res_fric:
            # If True enable friction and restitution on the walls
            v_x = self.velocity[0]
            v_y = self.velocity[1]
            v = np.linalg.norm(self.velocity)
            omega = self.omega

            if west_collision and not self.in_wall_collision[0]:
                self.in_wall_collision[0] = True
                v_x_1 = - self.e * v_x
                if self.cs or (- 2 / (7 * self.f * (self.e + 1)) < v_x / v and v_x / v < 0):
                    v_y_1 = np.sign(v_y) * (abs(v_y) - abs(self.f * (self.e + 1) * v_x))
                    omega_1 =  omega + np.sign(v_y) * abs(5 / (2 * self.radius) * self.f * (self.e + 1) * v_x)
                elif - 2 / (7 * self.f * (self.e + 1)) > v_x / v:
                    v_y_1 =  5/7 * (v_y + 2 * self.radius * self.omega / 5 )
                    omega_1 = v_y_1 / self.radius

                else:
                    msg = 'Error in west wall collision'
                    raise ValueError(msg)
                
                self.velocity = np.array([v_x_1, v_y_1])
                self.omega = omega_1
        
            if south_collision and not self.in_wall_collision[2]:
                self.in_wall_collision[2] = True
                v_y_1 = - self.e * v_y
                if  self.cs or (- 2 / (7 * self.f * (self.e + 1)) < v_y / v and v_y / v < 0):
                    v_x_1 = np.sign(v_x) * (abs(v_x) - abs(self.f * (self.e + 1) * v_y))
                    omega_1 =  omega - np.sign(v_x) * abs(5 / (2 * self.radius) * self.f * (self.e + 1) * v_y)
                elif - 2 / (7 * self.f * (self.e + 1)) > v_y / v:
                    v_x_1 =  5/7 * (v_x - 2 * self.radius * self.omega / 5 )
                    omega_1 = - v_x_1 / self.radius
                else:
                    msg = 'Error in south wall collision'
                    raise ValueError(msg)
                
                self.velocity = np.array([v_x_1, v_y_1])
                self.omega = omega_1

            if east_collision and not self.in_wall_collision[1]:
                self.in_wall_collision[1] = True
                v_x_1 = - self.e * v_x
                if self.cs or (2 / (7 * self.f * (self.e + 1)) > v_x / v and v_x / v > 0):
                    v_y_1 = np.sign(v_y) * (abs(v_y) - abs(self.f * (self.e + 1) * v_x))
                    omega_1 =  omega - np.sign(v_y) * abs(5 / (2 * self.radius) * self.f * (self.e + 1) * v_x)
                elif 2 / (7 * self.f * (self.e + 1)) < v_x / v:
                    v_y_1 =  5/7 * (v_y - 2 * self.radius * self.omega / 5 )
                    omega_1 = - v_y_1 / self.radius
                else:
                    msg = 'Error in east wall collision'
                    raise ValueError(msg)
                
                self.velocity = np.array([v_x_1, v_y_1])
                self.omega = omega_1

            if north_collision and not self.in_wall_collision[3]:
                self.in_wall_collision[3] = True
                v_y_1 = - self.e * v_y
                if self.cs or (2 / (7 * self.f * (self.e + 1)) > v_y / v and v_y / v > 0):
                    v_x_1 = np.sign(v_x) * (abs(v_x) - abs(self.f * (self.e + 1) * v_y))
                    omega_1 =  omega + np.sign(v_x) * abs(5 / (2 * self.radius) * self.f * (self.e + 1) * v_y)

                elif 2 / (7 * self.f * (self.e + 1)) < v_y / v:
                    v_x_1 =  5/7 * (v_x + 2 * self.radius * self.omega / 5 )
                    omega_1 =  v_x_1 / self.radius
                else:
                    msg = 'Error in north wall collision'
                    raise ValueError(msg)
                
                self.velocity = np.array([v_x_1, v_y_1])
                self.omega = omega_1

            self.in_wall_collision[0] = west_collision
            self.in_wall_collision[1] = east_collision
            self.in_wall_collision[2] = south_collision
            self.in_wall_collision[3] = north_collision
            
        else:
            # Else assume no friction and restitution on the walls
            if west_collision:
                self.velocity[0] = abs(self.velocity[0])
            if east_collision:
                self.velocity[0] = -abs(self.velocity[0])
            if south_collision:
                self.velocity[1] = abs(self.velocity[1])
            if north_collision:
                self.velocity[1] = -abs(self.velocity[1])

    @property
    def rotational_kinetic_energy(self):
        return 0.5 * self.I * self.omega ** 2
    
    @property
    def translational_kinetic_energy(self):
        return 0.5 * self.mass * np.linalg.norm(self.velocity) ** 2
    
    @property
    def kinetic_energy(self):
        return self.rotational_kinetic_energy + self.translational_kinetic_energy

class sim_particles():
    def __init__(self, window_size = np.array((800, 800)), nparticles = 20, e = 0.5, f = 0.5, random_particles = [True, True]):
        
        self.window_size = window_size
        self.nparticles = nparticles
        
        self.e = e
        self.f = f
        self.max_velocity = 200
        self.max_omega = 0.5

        self.max_rho = 1
        self.max_radius = 20

        self.rho_c = 1

        self.color1 = (20, 20, 250)
        self.color2 = (200, 200, 250)
        self.particles: list[particle] = []
        self.collisions = np.zeros((self.nparticles, self.nparticles))

        self.tau_pp_matrix = np.full((self.nparticles, self.nparticles), np.NaN)
        self.tau_pw_matrix = np.full(self.nparticles, np.NaN)

        self.window_size = window_size
        self.kinetic_energy = []
        self.velocity = []
        self.mean_free_path = []

        self.cs = self.f == 0 # use continuous sliding model if f = 0

        self.random_radius = random_particles[0]
        self.random_rho = random_particles[1]
        self.particle_res_fric = False # If True enable friction and restitution on the particles

        self.make_particles()
             
    def make_particles(self):
        # Make seed for reproducibility
        np.random.seed(1964)
        if self.random_radius:
            radius = np.random.rand(self.nparticles) * self.max_radius
        else:
            radius = np.full(self.nparticles, self.max_radius)

        if self.random_rho:
            rho = np.random.rand(self.nparticles) * self.max_rho
        else:
            rho = np.full(self.nparticles, self.max_rho)
        # make particles in a grid at least one diameter from walls  
        self.xstart =  np.ones(2) * self.max_radius * 2
        self.xlength = self.window_size - 2 * self.xstart
        # Check if the number of particles is too large for the window
        if self.nparticles > self.xlength[0] * self.xlength[1] / (np.pi * self.max_radius ** 2):
            raise ValueError('Number of particles is too large for the window')
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
            self.particles.append(particle(self.e, self.f, rho[i], self.rho_c, radius[i], position[i], velocity[i], 0, omega[i]))

    def make_particles_pyglet(self, batch):
        for i in range(self.nparticles):
            self.particles[i].circle = pyglet.shapes.Circle(self.particles[i].position[0], 
                                                            self.particles[i].position[1], 
                                                            self.particles[i].radius, color = self.color1, batch = batch)
            self.particles[i].circle2 = pyglet.shapes.Circle(self.particles[i].position[0] + 0.5 * self.particles[i].radius * np.cos(self.particles[i].angle), 
                                                            self.particles[i].position[1] + 0.5 * self.particles[i].radius * np.sin(self.particles[i].angle), 
                                                            0.25 * self.particles[i].radius, color = self.color2, batch = batch)
    
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
            x_i = np.append(self.particles[i].position,0)
            r_i = self.particles[i].radius
            for j in range(i+1, self.nparticles):
                x_j = np.append(self.particles[j].position,0)
                r_j = self.particles[j].radius
                distance = np.sqrt(((x_j - x_i)**2).sum())
                if  distance <= (r_i + r_j):
                    # collision! - apply textbook eq. 5.14
                    if self.collisions[i,j] == 1:
                        continue
                    self.collisions[i,j] = 1
                    
                    # Define variables for the i-th and j-th particle at the moment of contact
                    v_i = np.append(self.particles[i].velocity, 0)
                    omega_i = np.array([0, 0, self.particles[i].omega])
                    m_i = self.particles[i].mass

                    v_j = np.append(self.particles[j].velocity, 0)
                    omega_j = np.array([0, 0, self.particles[j].omega])
                    m_j = self.particles[j].mass

                    n = (x_j - x_i) / distance # unit vector between the i-th and j-th particle at the moment of contact
                    G0 = v_i - v_j # relative velocity at the moment of contact
                    
                    if self.particle_res_fric:
                        G0_c = G0 + np.cross(r_i*omega_i, n) + np.cross(r_j*omega_j, n) # relative velocity at the moment of contact in the contact frame
                        G0_ct = G0_c - np.dot(G0_c, n) * n # relative velocity at the moment of contact in the tangential frame

                        # Get the unit vector in the tangential frame from n
                        t_test = np.array([-n[1], n[0], 0])
                        t = G0_ct / np.linalg.norm(G0_ct) # unit vector in the tangential frame
                        if self.cs or np.dot(n, G0)/np.linalg.norm(G0_ct) < (2/7)*1/(self.f * (1 + self.e)): 
                            # Continuous sliding
                            v_i_1 = v_i - (n + self.f * t) * np.dot(n, G0) * (1 + self.e) * m_j / (m_i + m_j)
                            v_j_1 = v_j + (n + self.f * t) * np.dot(n, G0) * (1 + self.e) * m_i / (m_i + m_j)

                            omega_i_1 = omega_i - 5/(2*r_i) * np.dot(n, G0) * np.cross(n, t) * self.f * (1 + self.e) * m_j / (m_i + m_j)
                            omega_j_1 = omega_j - 5/(2*r_j) * np.dot(n, G0) * np.cross(n, t) * self.f * (1 + self.e) * m_i / (m_i + m_j)

                        else:
                            v_i_1 = v_i - ((1 + self.e) * np.dot(n, G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_j / (m_i + m_j)
                            v_j_1 = v_j + ((1 + self.e) * np.dot(n, G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_i / (m_i + m_j)

                            omega_i_1 = omega_i - 5/(7*r_i) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_j / (m_i + m_j)
                            omega_j_1 = omega_j - 5/(7*r_j) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_i / (m_i + m_j) 

                        self.particles[i].velocity = v_i_1[:2]
                        self.particles[j].velocity = v_j_1[:2]
                        self.particles[i].omega = omega_i_1[2]
                        self.particles[j].omega = omega_j_1[2]
                    
                    else:
                        # No friction and restitution on the particles but still account for the mass
                        dvel = n * np.dot(n, G0) 
                        self.particles[i].velocity -= dvel[:2] * 2 * m_j / (m_i + m_j)
                        self.particles[j].velocity += dvel[:2] * 2 * m_i / (m_i + m_j)

                    stop = True
                else:
                    self.collisions[i,j] = 0

    def time_before_collision_particle(self, particle_a: particle, particle_b: particle):
        r1 = particle_a.radius
        r2 = particle_b.radius
        
        x1 = particle_a.position
        x2 = particle_b.position
        v1 = particle_a.velocity
        v2 = particle_b.velocity
        
        delta_x = x2 - x1
        delta_v = v2 - v1

        if np.linalg.norm(delta_v) <= 0:
            return np.NaN
        
        a = np.dot(delta_v, delta_v)

        b = 2 * np.dot(delta_v, delta_x)
        if b >= 0:
            return np.NaN
        
        c = np.dot(delta_x, delta_x)

        sigma = r1 + r2
        
        discriminant = b ** 2 - 4 * a * (c - sigma ** 2)
        
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
        for i in range(self.nparticles-1):
            for j in range(i+1, self.nparticles):
                tau = self.time_before_collision_particle(self.particles[i], self.particles[j])
                self.tau_pp_matrix[j,i] = tau
        
        for i in range(self.nparticles):
            self.tau_pw_matrix[i] = self.time_before_collision_wall(self.particles[i])

    @property
    def dT_before_collision(self):
        return np.nanmin([np.nanmin(self.tau_pp_matrix), np.nanmin(self.tau_pw_matrix)])

    def get_total_kinetic_energy(self):
        return np.sum([particle.kinetic_energy for particle in self.particles])
    
    def get_velocities(self):
        return [np.linalg.norm(particle.velocity) for particle in self.particles]

    def get_mean_free_path(self):
        # mean free path is the average distance over which a moving particle travels before substantially changing its direction
        # Calculate the mean free path of the particles
        # Get the time before collision for each particle
        # Get the velocity of each particle
        # Get the distance travelled by each particle
        free_path = []
        for i in range(self.nparticles):
            tau_pp = np.nanmin(self.tau_pp_matrix[i])
            tau_pw = self.tau_pw_matrix[i]
            tau = np.nanmin([tau_pp, tau_pw])
            v = np.linalg.norm(self.particles[i].velocity)
            free_path.append(v * tau)
        return np.mean(free_path)

    def update_stats(self):
        self.kinetic_energy.append(self.get_total_kinetic_energy())
        self.velocity.append(self.get_velocities())
        self.mean_free_path.append(self.get_mean_free_path())

    def simulate(self, T, N):
        dt = T / N
        self.init_time_before_collision_matrix()
        
        self.t = np.arange(0, T, T/N)
        for i in range(N):
            print(f'Simulation time {i*dt:.2f} of {T}')
            self.move_particles(dt)
            self.wall_collision()
            self.particle_collision()
            self.init_time_before_collision_matrix()
            self.update_stats()
    
    def simulate_non_fixed_dt(self, T):
        t = 0
        self.init_time_before_collision_matrix()
        self.t = []
        while t < T:
            print(f'Simulation time {t:.2f} of {T}')
            dt = self.dT_before_collision + 1e-10
            self.move_particles(dt)
            self.wall_collision()
            self.particle_collision()
            self.init_time_before_collision_matrix()
            self.update_stats()
            self.t.append(t)
            t += dt

    def plot_kinetic_energy(self, T, N):
        plt.figure()
        plt.plot(self.t, self.kinetic_energy)
        plt.xlabel('time (s)')
        plt.ylabel('kinetic energy (J)')

    def plot_mean_free_path(self, T, N):
        plt.figure()
        plt.plot(np.arange(0, T, T/N), self.mean_free_path)
        plt.xlabel('time (s)')
        plt.ylabel('mean free path (m)')
    
    def plot_velocity_distribution(self, T, N, n_lst):
        dt = T/N
        plots = len(n_lst)
        fig, ax = plt.subplots(1, plots, figsize = (6 * plots, 4))
        for i in range(plots):
            ax[i].hist(self.velocity[n_lst[i]], bins = 20)
            ax[i].set_title(f'Velocity distribution at time {n_lst[i]*dt:.2f}')
            ax[i].set_xlabel('Velocity')
            ax[i].set_ylabel('Count')
        fig.tight_layout()

    def simulate_to_pyglet(self, dt):
        self.move_particles(dt)
        self.update_pyglet()
        self.wall_collision()
        self.particle_collision()

    def simulate_to_pyglet_non_fixed_dt(self, dt):
        self.init_time_before_collision_matrix()
        dt = self.dT_before_collision + 1e-10
        self.move_particles(dt)
        self.update_pyglet()
        self.wall_collision()
        self.particle_collision()

if __name__ == '__main__':
    _window_size = np.array((750, 750)) # window size
    _nparticles = 20 # number of particles
    _e = 1.0 # coefficient of restitution
    _f = 0 # coefficient of friction
    _random_particles = [False, False] # random radius and random rho
    sim = sim_particles(_window_size, _nparticles, _e, _f, _random_particles)

    if False:
    # if True:
        T = 20
        N = 500
        # sim.simulate(T, N)
        sim.simulate_non_fixed_dt(T)
        sim.plot_kinetic_energy(T, N)
        # sim.plot_mean_free_path(T, N)
        # sim.plot_velocity_distribution(T, N, [0, N-1])
        plt.show()
        stop = True

    else:
        window = pyglet.window.Window(sim.window_size[0], sim.window_size[1])
        batch = pyglet.graphics.Batch()

        @window.event
        def on_draw():
            window.clear()
            batch.draw()

        # pyglet.clock.schedule_interval(sim.simulate_to_pyglet, 1/120.0)
        pyglet.clock.schedule_interval(sim.simulate_to_pyglet_non_fixed_dt, 1/2)

        sim.make_particles_pyglet(batch)
        
        pyglet.app.run()

        del window
        del batch
            
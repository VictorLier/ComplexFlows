import numpy as np
import pyglet
import matplotlib.pyplot as plt

def grid(min_spacing: float, width: float, height: float, number: int) -> np.ndarray:
    '''
    Creates a grid of points with a minimum spacing between them
    min_spacing: minimum spacing between the points
    width: width of the grid
    height: height of the grid
    number: number of points
    '''
    # Reduced grid size
    x_start = min_spacing
    x_length = width - 3 * min_spacing
    y_length = height - 3 * min_spacing

    # Number of columns and rows that fits
    columns = int(np.ceil(x_length / min_spacing))
    rows = int(np.ceil(y_length / min_spacing))

    # Number of grid points
    grid_points = columns * rows


    if grid_points < number:
        '''
        Check if the grid is too small to create the number of points
        '''
        raise ValueError('The grid is too small to create the number of points')


    # The x values are found by each row
    x = np.zeros(columns)
    for i in range(columns):
        x[i] = x_start + i * min_spacing
    xg = np.tile(x, rows)

    # The y values are found by each column
    yg = np.zeros(columns*rows)
    for i in range(rows):
        yg[i*columns:(i+1)*columns] = np.repeat(i, columns) * min_spacing + min_spacing

    return xg, yg

class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, Rot: float, r: float, cd: float = 0.1, rho: float = 1):
        '''
        Creates a particle with the following parameters: x, y, vx, vy, Rot, r, cd, rho
        x, y: position
        vx, vy: velocity
        Rot: rotation
        r: radius
        cd: drag coefficient
        rho: density
        '''
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.Rot = Rot
        self.r = r
        self.cd = cd
        self.rho = rho
        self.circle = None
        self.circle2 = None
        self.angle = 0

        self.m = np.pi * self.r**2 * self.rho

        self.D = r*2

        self.I = 2/5 * self.m * self.r**2

    def speed(self):
        '''
        Returns the speed of the particle
        '''
        return np.sqrt(self.vx**2 + self.vy**2)
    
    def FD(self):
        '''
        Returns the drag force of the particle in the
        '''
        FD = 0.5 * self.speed()**2 * self.cd * self.D
        return FD
    
    def AD(self):
        '''
        Returns the drag acceleration of the particle in the x and y direction
        '''

        ADx = self.FD() / self.m * self.vx / self.speed()
        ADy = self.FD() / self.m * self.vy / self.speed()

        return ADx, ADy
    
    def dragSpeed(self, dt: float):
        '''
        Updates the speed of the particle with the drag force
        '''
        self.vx -= self.AD()[0]  * dt
        self.vy -= self.AD()[1]  * dt

    def Tke(self):
        '''
        Returns the translational kinetic energy of the particle
        '''
        return 0.5 * self.m * self.speed()**2
    
    def Rke(self):
        '''
        Returns the rotational kinetic energy of the particle
        '''
        return 0.5 * self.I * self.Rot**2
    
    def ke(self):
        '''
        Returns the kinetic energy of the particle
        '''
        return self.Tke() + self.Rke()
    
    def move(self, dt: float):
        '''
        Moves the particle
        dt: time to move
        '''
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.angle += self.Rot * dt
    
    def update_pyglet(self):
        '''
        Updates the pyglet circle
        '''
        self.circle.position = self.x, self.y
        self.circle2.position = self.x + 0.5 * self.r * np.cos(self.angle), self.y + 0.5 * self.r * np.sin(self.angle)


class FluidField:
    def __init__(self, x: float, y: float, N:int, Maxr: float, MaxV: float, MaxRot: float, e: float = 0.99, f: float = 0.4, TimeJump: bool = False, FricRes: bool = False):
        '''
        Defines the fluid field with the parameters: x, y, N, r, MaxV, MaxRot, e, f
        x, y: dimensions of the field
        N: number of particles
        r: radius of the particles
        MaxV: maximum velocity of the particles
        MaxRot: maximum rotation of the particles
        e: restitution coefficient
        f: friction coefficient
        TimeJump: True if the time jumps to next collision
        FricRes: True if friction and restitution is used
        '''
        self.x = x
        self.y = y
        self.N = N
        self.Maxr = Maxr
        self.MaxV = MaxV
        self.MaxRot = MaxRot
        self.e = e
        self.f = f
        self.TimeJump = TimeJump
        self.FricRes = FricRes


        # Create the particle starting point
        xg, yg = grid(3 * self.Maxr, self.x, self.y, self.N)

        # Create the empty particles list
        self.particles = np.empty(self.N, dtype=Particle)
        
        if True:
            # Velocity and rotation is randomized
            np.random.seed(0) # For reproducibility
            vx = np.random.rand(self.N) * self.MaxV
            vy = np.random.rand(self.N) * self.MaxV
            if self.FricRes:
                rot = np.random.rand(self.N) * self.MaxRot
            else:
                rot = np.random.rand(self.N) * 0
            r = np.random.rand(self.N) * self.Maxr

        else:
            # Velocity and rotation is constant
            vx = np.ones(self.N) * self.MaxV
            vy = np.ones(self.N) * self.MaxV
            if self.FricRes:
                rot = np.ones(self.N) * self.MaxRot
            else:
                rot = np.ones(self.N) * 0
            r = np.ones(self.N) * self.Maxr

        # Create the particles
        for i in range(self.N):
            self.particles[i] = Particle(xg[i], yg[i], vx[i], vy[i], rot[i], r[i])

        # Initialize the time matrix
        if self.TimeJump:
            self.Time = np.full((self.N + 4, self.N + 4), np.nan)

            # Finds the times
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        x1 = self.particles[i].x
                        y1 = self.particles[i].y
                        x2 = self.particles[j].x
                        y2 = self.particles[j].y
                        r1 = self.particles[i].r
                        r2 = r1
                        vx1 = self.particles[i].vx
                        vy1 = self.particles[i].vy
                        vx2 = self.particles[j].vx
                        vy2 = self.particles[j].vy

                        # Find the time to the other particle
                        t = (-vx1*x1 + vx1*x2 + vx2*x1 - vx2*x2 - vy1*y1 + vy1*y2 + vy2*y1 - vy2*y2 - np.sqrt(-vx1**2*y1**2 + 2*vx1**2*y1*y2 - vx1**2*y2**2 + vx1**2*(r1 + r2)**2 + 2*vx1*vx2*y1**2 - 4*vx1*vx2*y1*y2 + 2*vx1*vx2*y2**2 - 2*vx1*vx2*(r1 + r2)**2 + 2*vx1*vy1*x1*y1 - 2*vx1*vy1*x1*y2 - 2*vx1*vy1*x2*y1 + 2*vx1*vy1*x2*y2 - 2*vx1*vy2*x1*y1 + 2*vx1*vy2*x1*y2 + 2*vx1*vy2*x2*y1 - 2*vx1*vy2*x2*y2 - vx2**2*y1**2 + 2*vx2**2*y1*y2 - vx2**2*y2**2 + vx2**2*(r1 + r2)**2 - 2*vx2*vy1*x1*y1 + 2*vx2*vy1*x1*y2 + 2*vx2*vy1*x2*y1 - 2*vx2*vy1*x2*y2 + 2*vx2*vy2*x1*y1 - 2*vx2*vy2*x1*y2 - 2*vx2*vy2*x2*y1 + 2*vx2*vy2*x2*y2 - vy1**2*x1**2 + 2*vy1**2*x1*x2 - vy1**2*x2**2 + vy1**2*(r1 + r2)**2 + 2*vy1*vy2*x1**2 - 4*vy1*vy2*x1*x2 + 2*vy1*vy2*x2**2 - 2*vy1*vy2*(r1 + r2)**2 - vy2**2*x1**2 + 2*vy2**2*x1*x2 - vy2**2*x2**2 + vy2**2*(r1 + r2)**2))/(vx1**2 - 2*vx1*vx2 + vx2**2 + vy1**2 - 2*vy1*vy2 + vy2**2)
                        if t < 0:
                            t = np.nan
                        self.Time[i,j] = t

            # Find the time to the walls
            for i in range(self.N):
                t_south = (self.particles[i].r - self.particles[i].y) / self.particles[i].vy
                if t_south < 0:
                    t_south = np.nan
                self.Time[i, -4] = t_south

                t_west = (self.particles[i].r - self.particles[i].x) / self.particles[i].vx
                if t_west < 0:
                    t_west = np.nan
                self.Time[i, -3] = t_west

                t_north = (y - self.particles[i].r - self.particles[i].y) / self.particles[i].vy
                if t_north < 0:
                    t_north = np.nan
                self.Time[i, -2] = t_north

                t_east = (x - self.particles[i].r - self.particles[i].x) / self.particles[i].vx
                if t_east < 0:
                    t_east = np.nan
                self.Time[i, -1] = t_east

    def time_to_collision(self):
        '''
        Finds the time to collision and updates the time vector
        '''
        dt = np.nanmin(self.Time)
        index = np.where(self.Time == dt)
        index = [index[0][0], index[1][0]]
        self.Time = self.Time - dt
        return dt, index

    def update_time(self, index):
        '''
        Updates the time matrix
        index: index of the collision
        '''
        # Particle particle
        for index, i in enumerate(index):
            for j in range(self.N):
                if i != j:
                    x1 = self.particles[i].x
                    y1 = self.particles[i].y
                    x2 = self.particles[j].x
                    y2 = self.particles[j].y
                    r1 = self.particles[i].r
                    r2 = r1
                    vx1 = self.particles[i].vx
                    vy1 = self.particles[i].vy
                    vx2 = self.particles[j].vx
                    vy2 = self.particles[j].vy
                    # Find the time to the other particle
                    t = (-vx1*x1 + vx1*x2 + vx2*x1 - vx2*x2 - vy1*y1 + vy1*y2 + vy2*y1 - vy2*y2 - np.sqrt(-vx1**2*y1**2 + 2*vx1**2*y1*y2 - vx1**2*y2**2 + vx1**2*(r1 + r2)**2 + 2*vx1*vx2*y1**2 - 4*vx1*vx2*y1*y2 + 2*vx1*vx2*y2**2 - 2*vx1*vx2*(r1 + r2)**2 + 2*vx1*vy1*x1*y1 - 2*vx1*vy1*x1*y2 - 2*vx1*vy1*x2*y1 + 2*vx1*vy1*x2*y2 - 2*vx1*vy2*x1*y1 + 2*vx1*vy2*x1*y2 + 2*vx1*vy2*x2*y1 - 2*vx1*vy2*x2*y2 - vx2**2*y1**2 + 2*vx2**2*y1*y2 - vx2**2*y2**2 + vx2**2*(r1 + r2)**2 - 2*vx2*vy1*x1*y1 + 2*vx2*vy1*x1*y2 + 2*vx2*vy1*x2*y1 - 2*vx2*vy1*x2*y2 + 2*vx2*vy2*x1*y1 - 2*vx2*vy2*x1*y2 - 2*vx2*vy2*x2*y1 + 2*vx2*vy2*x2*y2 - vy1**2*x1**2 + 2*vy1**2*x1*x2 - vy1**2*x2**2 + vy1**2*(r1 + r2)**2 + 2*vy1*vy2*x1**2 - 4*vy1*vy2*x1*x2 + 2*vy1*vy2*x2**2 - 2*vy1*vy2*(r1 + r2)**2 - vy2**2*x1**2 + 2*vy2**2*x1*x2 - vy2**2*x2**2 + vy2**2*(r1 + r2)**2))/(vx1**2 - 2*vx1*vx2 + vx2**2 + vy1**2 - 2*vy1*vy2 + vy2**2)
                    if t < 0:
                        t = np.nan
                    self.Time[i,j] = t

            for index, i in enumerate(index):
                t_south = (self.particles[i].r - self.particles[i].y) / self.particles[i].vy
                if t_south < 0:
                    t_south = np.nan
                self.Time[i, -4] = t_south

                t_west = (self.particles[i].r - self.particles[i].x) / self.particles[i].vx
                if t_west < 0:
                    t_west = np.nan
                self.Time[i, -3] = t_west

                t_north = (y - self.particles[i].r - self.particles[i].y) / self.particles[i].vy
                if t_north < 0:
                    t_north = np.nan
                self.Time[i, -2] = t_north

                t_east = (x - self.particles[i].r - self.particles[i].x) / self.particles[i].vx
                if t_east < 0:
                    t_east = np.nan
                self.Time[i, -1] = t_east
        
    def make_pyglet_particles(self, batch: pyglet.graphics.Batch):
        '''
        Makes the pyglet particles
        '''
        for i in range(self.N):
            self.particles[i].circle = pyglet.shapes.Circle(self.particles[i].x, self.particles[i].y, self.particles[i].r, color=(153, 0, 0), batch=batch)
            self.particles[i].circle2 = pyglet.shapes.Circle(self.particles[i].x + 0.5 * self.particles[i].r * np.cos(self.particles[i].angle), self.particles[i].y + 0.5 * self.particles[i].r * np.sin(self.particles[i].angle), 0.25*self.particles[i].r, color=(255, 255, 255), batch=batch)

    def move_all(self, dt: float):
        '''
        Moves all particles in the fluid field
        dt: time to move
        '''
        if False: # add drag
            for i in range(self.N):
                self.particles[i].dragSpeed(dt)

        for i in range(self.N):
            self.particles[i].move(dt)
    
    def update_pyglet(self):
        '''
        Updates the pyglet particles
        '''
        for i in range(self.N):
            self.particles[i].update_pyglet()

    def collision(self):
        '''
        Checks for collisions and updates the particle velocities
        '''
        # Wall collisions
        wall_overlap = np.full((self.N,4), False)
        for i in range(self.N):
            if self.particles[i].x - self.particles[i].r < 0:
                if wall_overlap[i, 0]:
                    continue
                self.particles[i].vx = abs(self.particles[i].vx)
                wall_overlap[i, 0] = True
            else:
                wall_overlap[i, 0] = False
            if self.particles[i].x + self.particles[i].r > self.x:
                if wall_overlap[i, 1]:
                    continue
                self.particles[i].vx = -abs(self.particles[i].vx)
                wall_overlap[i, 1] = True
            else:
                wall_overlap[i, 1] = False
            if self.particles[i].y - self.particles[i].r < 0:
                if wall_overlap[i, 2]:
                    continue
                self.particles[i].vy = abs(self.particles[i].vy)
                wall_overlap[i, 2] = True
            else:
                wall_overlap[i, 2] = False
            if self.particles[i].y + self.particles[i].r > self.y:
                if wall_overlap[i, 3]:
                    continue
                self.particles[i].vy = -abs(self.particles[i].vy)
                wall_overlap[i, 3] = True
            else:
                wall_overlap[i, 3] = False
        
        # Particle collisions
        # Initialize the overlap matrix
        overlap = np.full((self.N, self.N), False)

        if self.FricRes: # Friction and restitution
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    if i == j:
                        continue
                    
                    x_i = np.array([self.particles[i].x, self.particles[i].y, 0])
                    x_j = np.array([self.particles[j].x, self.particles[j].y, 0])
                    r_i = self.particles[i].r
                    r_j = self.particles[j].r

                    distance = np.sqrt(((x_j - x_i)**2).sum())
                    # Check if the particles are overlapping
                    if distance < (r_i + r_j): # If the particles are overlapping
                        if overlap[i, j]: # If the particles are already overlapping
                            continue
                        overlap[i, j] = True # Set the overlap to True
                        v_i = np.array([self.particles[i].vx, self.particles[i].vy, 0])
                        omega_i = np.array([0, 0, self.particles[i].Rot])
                        m_i = self.particles[i].m

                        v_j = np.array([self.particles[j].vx, self.particles[j].vy, 0])
                        omega_j = np.array([0, 0, self.particles[j].Rot])   
                        m_j = self.particles[j].m

                        n = (x_j - x_i) / distance
                        G0 = v_i - v_j

                        G0_c = G0 + np.cross(r_i * omega_i, n) + np.cross(r_j * omega_j, n)
                        G0_ct = G0_c -np.dot(G0_c, n) * n
                        
                        t = G0_ct / np.linalg.norm(G0_ct)

                        if np.dot(n, G0) / np.linalg.norm(G0_ct) >= (2/7) * 1 / (self.f * (1 + self.e)): # Check for continus sliding
                            v1 = v_i - (n + self.f*t) * np.dot(n, G0) * (1 + self.e) * m_j / (m_j + m_i)
                            v2 = v_j + (n + self.f*t) * np.dot(n, G0) * (1 + self.e) * m_i / (m_j + m_i)
                            omega1 = omega_i - 5/(2*r_i) * np.dot(n, G0) * np.cross(n, t) * self.f*(1 + self.e) * m_j / (m_j + m_i)
                            omega2 = omega_j - 5/(2*r_j) * np.dot(n, G0) * np.cross(n, t) * self.f*(1 + self.e) * m_i / (m_j + m_i)

                        else:
                            v1 = v_i - ((1 + self.e) * np.dot(n,G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_j / (m_j + m_i)
                            v2 = v_j + ((1 + self.e) * np.dot(n,G0) * n + 2/7 * np.linalg.norm(G0_ct) * t) * m_i / (m_j + m_i)
                            omega1 = omega_i - 5/(7*r_i) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_j / (m_j + m_i)
                            omega2 = omega_j - 5/(7*r_j) * np.linalg.norm(G0_ct) * np.cross(n, t) * m_i / (m_j + m_i)

                        self.particles[i].vx = v1[0]
                        self.particles[i].vy = v1[1]
                        self.particles[j].vx = v2[0]
                        self.particles[j].vy = v2[1]
                        self.particles[i].Rot = omega1[2]
                        self.particles[j].Rot = omega2[2]
                        
                    else:
                        overlap[i, j] = False
        
        else: # No friction and restitution
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    if i == j:
                        continue
                    # Check if the particles are overlapping
                    distance = np.sqrt((self.particles[j].x - self.particles[i].x)**2 + (self.particles[j].y - self.particles[i].y)**2)
                    if distance < self.particles[i].r + self.particles[j].r: # If the particles are overlapping
                        if overlap[i, j]: # If the particles are already overlapping
                            continue
                        overlap[i, j] = True # Set the overlap to True
                        G0 = np.array([self.particles[i].vx - self.particles[j].vx, self.particles[i].vy - self.particles[j].vy])
                        n = np.array([self.particles[j].x - self.particles[i].x, self.particles[j].y - self.particles[i].y]) / distance
                        dvel = n * np.dot(n, G0)
                        self.particles[i].vx -= dvel[0] * 2 * self.particles[j].m / (self.particles[i].m + self.particles[j].m)
                        self.particles[i].vy -= dvel[1] * 2 * self.particles[j].m / (self.particles[i].m + self.particles[j].m)
                        self.particles[j].vx += dvel[0] * 2 * self.particles[i].m / (self.particles[i].m + self.particles[j].m)
                        self.particles[j].vy += dvel[1] * 2 * self.particles[i].m / (self.particles[i].m + self.particles[j].m)
                        
                    else:
                        overlap[i, j] = False

    def ke_total(self):
        '''
        Returns the kinetic energy of the fluid field
        '''
        kinetic = 0
        for i in range(self.N):
            kinetic += self.particles[i].ke()
        return kinetic
    
    def speed_total(self):
        '''
        Returns the total speed of the fluid field
        '''
        speed = 0
        for i in range(self.N):
            speed += self.particles[i].speed()
        return speed

    def do_in_sim(self):
        '''
        Things to do in the sim
        '''
        print(self.ke_total())
        print(self.speed_total())

    def simulate(self, dt: float, t: float):
        '''
        Simulates the fluid field for a given time
        dt: time step
        t: time to simulate
        '''
        time = 0
        if self.TimeJump:
            while time < t:
                self.do_in_sim()
                dt, index = self.time_to_collision()
                self.move_all(dt)
                self.collision()
                self.update_time(index)
                time += dt
        
        else:
            while time < t:
                self.do_in_sim()
                self.move_all(dt)
                self.collision()
                time += dt
            
    def simulate_pyglet(self, dt: float):
        '''
        Simulates the fluid field in pyglet
        dt: time step
        '''
        self.move_all(dt)
        self.update_pyglet()
        self.collision()

if __name__ == "__main__":
    field = FluidField(x=1000, y=1000, N=30, Maxr=40, MaxV=100, MaxRot=10,TimeJump=False, FricRes=False) 

    if False: # simulate
        field.simulate(0.01, 10)
    
    
    if True: # Pyglet
        window = pyglet.window.Window(1000, 1000)
        batch = pyglet.graphics.Batch()
        @window.event
        def on_draw():
            window.clear()
            batch.draw()
        pyglet.clock.schedule_interval(field.simulate_pyglet, 1/30)
        field.make_pyglet_particles(batch)
        pyglet.app.run()
        del window
        del batch    


    print("Stop")
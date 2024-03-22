import numpy as np

def grid(min_spacing: float, width: float, height: float, number: int) -> np.ndarray:
    '''
    Creates a grid of points with a minimum spacing between them
    min_spacing: minimum spacing between the points
    width: width of the grid
    number: number of points
    '''
    # Reduced grid size
    x_start = min_spacing
    x_length = width - 2 * min_spacing
    y_length = height - 2 * min_spacing

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


def time_map(N: int) -> np.ndarray:
    '''
    Creates a map of the time vector and the time vector
    N: number of particles
    out: t_map, t_vector
    '''
    # Expand matrix size to include walls
    n = N + 4

    # Finde the number of elements in the time vector
    number = 0
    for i in range(N):
        number += n-(i+1)
    
    # Create the empty time vector
    t_vector = np.zeros(number)

    # Create the time vector
    time_vector = np.arange(number)
    # Add the wall elements
    time_vector_wall = np.append(time_vector, [-99999, -99999, -99999, -99999, -99999, -99999])

    # Create the time map
    t_map = np.full((n, n), -99999)

    # Finds the indices of the upper triangle of the matrix
    inds = np.triu_indices_from(t_map, k=1)

    # Fills the upper and lower triangle with the time vector
    t_map[inds] = time_vector_wall
    t_map.T[inds] = time_vector_wall

    return t_map, t_vector


def time_to_collision(Particles, time_map, time_vector, x, y):
    '''
    Finds the time to collision for each particle
    Particles: array of particles
    time_map: map of the time vector
    time_vector: time vector
    x: domain x
    y: domain y
    '''
    # Find the time to the other particle
    for i in range(len(Particles)):
        for j in range(len(Particles)):
            if i != j:
                x1 = Particles[i].x
                y1 = Particles[i].y
                x2 = Particles[j].x
                y2 = Particles[j].y
                r1 = Particles[i].r
                r2 = r1
                vx1 = Particles[i].vx
                vy1 = Particles[i].vy
                vx2 = Particles[j].vx
                vy2 = Particles[j].vy

                # Find the time to the other particle
                t = (-vx1*x1 + vx1*x2 + vx2*x1 - vx2*x2 - vy1*y1 + vy1*y2 + vy2*y1 - vy2*y2 - np.sqrt(-vx1**2*y1**2 + 2*vx1**2*y1*y2 - vx1**2*y2**2 + vx1**2*(r1 + r2)**2 + 2*vx1*vx2*y1**2 - 4*vx1*vx2*y1*y2 + 2*vx1*vx2*y2**2 - 2*vx1*vx2*(r1 + r2)**2 + 2*vx1*vy1*x1*y1 - 2*vx1*vy1*x1*y2 - 2*vx1*vy1*x2*y1 + 2*vx1*vy1*x2*y2 - 2*vx1*vy2*x1*y1 + 2*vx1*vy2*x1*y2 + 2*vx1*vy2*x2*y1 - 2*vx1*vy2*x2*y2 - vx2**2*y1**2 + 2*vx2**2*y1*y2 - vx2**2*y2**2 + vx2**2*(r1 + r2)**2 - 2*vx2*vy1*x1*y1 + 2*vx2*vy1*x1*y2 + 2*vx2*vy1*x2*y1 - 2*vx2*vy1*x2*y2 + 2*vx2*vy2*x1*y1 - 2*vx2*vy2*x1*y2 - 2*vx2*vy2*x2*y1 + 2*vx2*vy2*x2*y2 - vy1**2*x1**2 + 2*vy1**2*x1*x2 - vy1**2*x2**2 + vy1**2*(r1 + r2)**2 + 2*vy1*vy2*x1**2 - 4*vy1*vy2*x1*x2 + 2*vy1*vy2*x2**2 - 2*vy1*vy2*(r1 + r2)**2 - vy2**2*x1**2 + 2*vy2**2*x1*x2 - vy2**2*x2**2 + vy2**2*(r1 + r2)**2))/(vx1**2 - 2*vx1*vx2 + vx2**2 + vy1**2 - 2*vy1*vy2 + vy2**2)
                if t < 0:
                    t = np.nan
                time_vector[time_map[i, j]] = t

    # Find the time to the walls
    for i in range(len(Particles)):
        t_south = (Particles[i].r - Particles[i].y) / Particles[i].vy
        if t_south < 0:
            t_south = np.nan
        time_vector[time_map[i, -4]] = t_south

        t_west = (Particles[i].r - Particles[i].x) / Particles[i].vx
        if t_west < 0:
            t_west = np.nan
        time_vector[time_map[i, -3]] = t_west

        t_north = (y - Particles[i].r - Particles[i].y) / Particles[i].vy
        if t_north < 0:
            t_north = np.nan
        time_vector[time_map[i, -2]] = t_north

        t_east = (x - Particles[i].r - Particles[i].x) / Particles[i].vx
        if t_east < 0:
            t_east = np.nan
        time_vector[time_map[i, -1]] = t_east

    return time_vector


def move(fluidfield, time: float):
    '''
    Moves the particles in the fluid field
    fluidfield: fluid field
    time: time to move
    out: fluidfield
    '''
    for i in range(fluidfield.N):
        fluidfield.particles[i].x += fluidfield.particles[i].vx * time
        fluidfield.particles[i].y += fluidfield.particles[i].vy * time

    return fluidfield


def collision(particle1, particle2):
    n = np.array([particle2.x - particle1.x, particle2.y - particle1.y]) / particle1.r
    G0 = np.array([particle1.vx - particle2.vx, particle1.vy - particle2.vy])
    dvel = n * np.dot(n, G0)
    particle1.vx -= dvel[0]
    particle1.vy -= dvel[1]
    particle2.vx += dvel[0]
    particle2.vy += dvel[1]


class Particle:
    def __init__(self, x: float, y: float, vx: float, vy: float, Rot: float, r: float):
        '''
        Creates a particle with the following parameters: x, y, vx, vy, Rot, r
        x, y: position
        vx, vy: velocity
        Rot: rotation
        r: radius
        '''
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.Rot = Rot
        self.r = r

    def speed(self):
        '''
        Returns the speed of the particle
        '''
        return np.sqrt(self.vx**2 + self.vy**2)
    
    def ke(self):
        '''
        Returns the kinetic energy of the particle
        '''
        return 0.5 * self.speed()**2


class FluidField:
    def __init__(self, x: float, y: float, N:int,r: float, MaxV: float, MaxRot: float):
        '''
        Defines the fluid field with the parameters: x, y, N, r
        x, y: dimensions of the field
        N: number of particles
        r: radius of the particles
        MaxV: maximum velocity of the particles
        MaxRot: maximum rotation of the particles
        '''
        self.x = x
        self.y = y
        self.N = N
        self.r = r
        self.MaxV = MaxV
        self.MaxRot = MaxRot

        # Create the particle starting point
        xg, yg = grid(3 * r, self.x, self.y, self.N)

        # Create the empty particles list
        self.particles = np.empty(self.N, dtype=Particle)
        
        # Velocity and rotation is randomized
        vx = np.random.rand(self.N) * self.MaxV
        vy = np.random.rand(self.N) * self.MaxV
        rot = np.random.rand(self.N) * self.MaxRot

        # Create the particles
        for i in range(self.N):
            self.particles[i] = Particle(xg[i], yg[i], vx[i], vy[i], rot[i], self.r)

        # Create the time map and vector
        self.time_map, self.time_vector = time_map(self.N)

        # Find the time to collision
        self.time_vector = time_to_collision(self.particles, self.time_map, self.time_vector, self.x, self.y)

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
    
    def get_lowest_time(self):
        '''
        Returns the lowest time and index of the time vector
        out: min_value, min_index
        '''
        min_value = np.nanmin(self.time_vector)
        min_index = np.nanargmin(self.time_vector)

        return min_value, min_index

    def get_next_pair(self):
        '''
        Returns the next pair of particles to collide
        out: index
        '''
        min_index = self.get_lowest_time()[1]
        index = np.where(self.time_map == min_index)[0]

        return index

class Simulation:
    def __init__(self, total_time: float, field: FluidField):
        '''
        Creates a simulation with the parameters: dt, total_time, field
        total_time: total time of the simulation
        field: fluid field
        '''
        self.total_time = total_time
        self.field = field

    def run_steps(self, steps: int):
        next_time = 0
        for i in range(steps):
            next_time = self.field.get_lowest_time()[0]
            self.field = move(self.field, next_time)


if __name__ == "__main__":

    field = FluidField(x = 10, y = 10, N = 4, r = 1, MaxV = 1, MaxRot = 1)

    sim = Simulation(total_time=2, field=field)
    sim.run_steps(1)

    print("Stop")
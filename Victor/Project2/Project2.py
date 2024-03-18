import numpy as np

def grid(min_spacing: float, width: float, height: float, number: int):
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

        # Check if it is possible to create the particle positions
        if (x < 3 * r) or (y < 3 * r):
            raise ValueError('The field is too small to create the particles')


        # Create the particle starting point
        xg, yg = grid(3 * r, self.x, self.y, self.N)

        # Create the particles
        self.particles = np.empty(self.N, dtype=Particle)
        
        # Velocity and rotation is randomized
        vx = np.random.rand(self.N) * self.MaxV
        vy = np.random.rand(self.N) * self.MaxV
        rot = np.random.rand(self.N) * self.MaxRot

        for i in range(self.N):
            self.particles[i] = Particle(xg[i], yg[i], vx[i], vy[i], rot[i], self.r)


    def ke_total(self):
        '''
        Returns the kinetic energy of the fluid field
        '''
        kinetic = 0
        for i in range(self.N):
            kinetic += self.particles[i].ke()
        return kinetic


if __name__ == "__main__":

    field = FluidField(10, 10, 4, 1, 1, 1)


    print("Stop")
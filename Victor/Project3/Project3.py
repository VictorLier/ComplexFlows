import numpy as np

def latex_plot(x: np.ndarray, y: np.ndarray, name: str, folder = "Victor/Project3/plots"):
    '''
    Creates the data file for plotting in latax:
    name: name of the file
    x: x values
    y: y values
    '''
    # Check if the x and y values have the same length
    if len(x) != len(y):
        raise ValueError("latex_plot: The x and y values must have the same length")
    
    file_name = (f"{folder}/{name}.txt")
    np.savetxt(file_name, np.array([x, y]).T, fmt = ["%f", "%f"])

class Capillary():
    def __init__(self, H: float = 0.001, L: float = 0.01, theta: float = 20, rho: float = 997, mu: float = 8.88e-4) -> None:
        '''
        Capillary class
        H: height of the capillary [m]
        L: length of the capillary [m]
        theta: contact angle [degrees]
        rho: density of the liquid [kg/m^3]
        mu: viscosity of the liquid [Pa s]
        '''
        self.H = H
        self.L = L
        self.theta = np.radians(theta)
        self.rho = rho
        self.mu = mu


    


if __name__ == "__main__":
    if True: #Test
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        latex_plot(x, y, "test")

    if True: # Question 2
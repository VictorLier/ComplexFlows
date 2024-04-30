import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

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

def load_data(name: str, folder:str="Victor/Project3/cfd_results") -> np.ndarray:
    '''
    Loads data from a file
    returns:
    x: x values
    y: y values
    '''
    file_name = (f"{folder}/{name}.csv")
    data = np.loadtxt(file_name, skiprows=1, delimiter=",")
    return data[:, 0], data[:, 1]


class Capillary():
    def __init__(self, H:float=0.001, L:float=0.01, theta:float=20, rho_l:float=997, rho_g:float=1, mu_l:float=8.88e-4, mu_g:float=17.9e-6, gamma:float=0.072) -> None:
        '''
        Capillary class
        H: height of the capillary [m]
        L: length of the capillary [m]
        theta: contact angle [degrees]
        rho_l: density of the liquid [kg/m^3]
        rho_g: density of the gas [kg/m^3]
        mu_l: dynamic viscosity of the liquid [Pa s]
        mu_g: dynamic viscosity of the gas [Pa s]
        gamma: surface tension [N/m]
        '''
        self.H = H
        self.L = L
        self.theta = np.radians(theta)
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.mu_g = mu_g
        self.gamma = gamma
    
    def time(self, T=None) -> np.ndarray:
        '''
        Creates the time array
        '''
        if T is None:
            T = 0.02
        self.t = np.linspace(0, T, 100)

    def normal_noslip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the normal noslip CFD simulation
        '''
        x, y = load_data("normal_noslip_timehistory")

        if plot:
            plt.plot(x, y, label="normal_noslip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "normal_noslip_timehistory")

    def normal_noslip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the normal noslip CFD simulation
        '''
        x, y = load_data("normal_noslip_velocityprofile")

        if plot:
            plt.plot(x, y, label="normal_noslip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "normal_noslip_velocityprofile")

    def normal_slip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the normal slip CFD simulation
        '''
        x, y = load_data("normal_slip_timehistory")

        if plot:
            plt.plot(x, y, label="normal_slip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "normal_slip_timehistory")
    
    def normal_slip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the normal slip CFD simulation
        '''
        x, y = load_data("normal_slip_velocityprofile")

        if plot:
            plt.plot(x, y, label="normal_slip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "normal_slip_velocityprofile")

    def fillet_noslip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the fillet noslip CFD simulation
        '''
        x, y = load_data("fillet_noslip_timehistory")

        if plot:
            plt.plot(x, y, label="fillet_noslip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_timehistory")
    
    def fillet_noslip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the fillet noslip CFD simulation
        '''
        x, y = load_data("fillet_noslip_velocityprofile")

        if plot:
            plt.plot(x, y, label="fillet_noslip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_velocityprofile")
    
    def fillet_slip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the fillet slip CFD simulation
        '''
        x, y = load_data("fillet_slip_timehistory")

        if plot:
            plt.plot(x, y, label="fillet_slip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "fillet_slip_timehistory")
    
    def fillet_slip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the fillet slip CFD simulation
        '''
        x, y = load_data("fillet_slip_velocityprofile")

        if plot:
            plt.plot(x, y, label="fillet_slip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "fillet_slip_velocityprofile")

    def waterwater_noslip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the water-water noslip CFD simulation
        '''
        x, y = load_data("waterwater_noslip_timehistory")

        if plot:
            plt.plot(x, y, label="waterwater_noslip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "waterwater_noslip_timehistory")
    
    def waterwater_noslip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the water-water noslip CFD simulation
        '''
        x, y = load_data("waterwater_noslip_velocityprofile")

        if plot:
            plt.plot(x, y, label="waterwater_noslip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "waterwater_noslip_velocityprofile")
    
    def waterwater_slip_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the water-water slip CFD simulation
        '''
        x, y = load_data("waterwater_slip_timehistory")

        if plot:
            plt.plot(x, y, label="waterwater_slip_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "waterwater_slip_timehistory")
    
    def waterwater_slip_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the water-water slip CFD simulation
        '''
        x, y = load_data("waterwater_slip_velocityprofile")

        if plot:
            plt.plot(x, y, label="waterwater_slip_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "waterwater_slip_velocityprofile")

    def fillet_noslip_closed_comp_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the fillet noslip closed compartment CFD simulation
        '''
        x, y = load_data("fillet_noslip_closed_comp_timehistory")

        if plot:
            plt.plot(x, y, label="fillet_noslip_closed_comp_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_closed_comp_timehistory")

    def fillet_noslip_closed_comp_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the fillet noslip closed compartment CFD simulation
        '''
        x, y = load_data("fillet_noslip_closed_comp_velocityprofile")

        if plot:
            plt.plot(x, y, label="fillet_noslip_closed_comp_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_closed_comp_velocityprofile")
    
    def fillet_noslip_closed_uncomp_timehistory(self, plot:bool=False) -> None:
        '''
        Loads the time history from the fillet noslip open compartment CFD simulation
        '''
        x, y = load_data("fillet_noslip_closed_uncomp_timehistory")

        if plot:
            plt.plot(x, y, label="fillet_noslip_closed_uncomp_timehistory")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_closed_uncomp_timehistory")
    
    def fillet_noslip_closed_uncomp_velocityprofile(self, plot:bool=False) -> None:
        '''
        Loads the velocity profile from the fillet noslip open compartment CFD simulation
        '''
        x, y = load_data("fillet_noslip_closed_uncomp_velocityprofile")

        if plot:
            plt.plot(x, y, label="fillet_noslip_closed_uncomp_velocityprofile")
            plt.xlabel("Distance [m]")
            plt.ylabel("Velocity [m/s]")
            plt.legend()

            latex_plot(x, y, "fillet_noslip_closed_uncomp_velocityprofile")

    def Lucas_washburn(self, plot:bool=False) -> None:
        '''
        Lucas-Washburn equation
        '''
        if "self.t" not in locals():
            self.time()
        self.lucas_washburn = np.sqrt(self.H * self.gamma * np.cos(self.theta) / (3 * self.mu_l) * self.t)

        if plot:
            plt.plot(self.t, self.lucas_washburn, label="lucas_washburn")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(self.t, self.lucas_washburn, "lucas_washburn")   

    def Bosanquet(self, plot:bool=False) -> None:
        '''
        Bosanquet equation
        '''
        if "self.t" not in locals():
            self.time()

        A_I = np.sqrt(2 * self.gamma * np.cos(self.theta) / (self.rho_l * self.H))
        B = 12 * self.mu_l / (self.rho_l * self.H**2)

        self.bosanquet = np.sqrt(2 * A_I**2 / B * (self.t - 1 / B * (1 - np.exp(-B * self.t))))

        if plot:
            plt.plot(self.t, self.bosanquet, label="bosanquet")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(self.t, self.bosanquet, "bosanquet")
    
    def Lucas_washburn_modified(self, plot:bool=False) -> None:
        '''
        Lucas_washburn modified equation
        '''
        if "self.t" not in locals():
            self.time()
        
        h = self.H/2

        Alpha = self.mu_g * self.L / (self.mu_l * h)
        t_tilde = self.t * self.gamma * np.cos(self.theta) / (6 * self.mu_l * h)
        self.lucas_washburn_modified = (np.sqrt(t_tilde + Alpha**2) - Alpha) * h
        
        if plot:
            plt.plot(self.t, self.lucas_washburn_modified, label="lucas_washburn_modified")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(self.t, self.lucas_washburn, "lucas_washburn_modified")

    def Bosanquet_modified(self, plot:bool=False) -> None:
        '''
        Bosanquet modified equation
        '''
        if "self.t" not in locals():
            self.time()

        # Define the variables
        L, B, A, t = sp.symbols('L B A t')

        # Define the function
        l = sp.Function('l')(t)
        eq1 = sp.Eq(L * sp.Derivative(l,t,t) + B * L * sp.Derivative(l,t), A**2)
        ics = {l.subs(t,0): 0, sp.Derivative(l,t).subs(t,0): 0}

        # Solve the differential equation
        sol = sp.dsolve(eq1, ics=ics)

        # Get the function l(t)
        l_function = sol.rhs  # Extract the right-hand side of the solution

        # Input the values of A, B, L, and t into the solution
        A_ = sp.sqrt(2 * self.gamma * sp.cos(self.theta) / (self.rho_l * self.H))
        B_ = 12 * self.mu_l / (self.rho_l * self.H**2)

        # Substitute the values into the function
        l_func = sp.lambdify(t, l_function.subs({A: A_, B: B_, L: self.L}), modules="numpy")
        self.bosanquet_modified = l_func(self.t)


        if plot:
            plt.plot(self.t, self.bosanquet_modified, label="bosanquet_modified")
            plt.xlabel("Time [s]")
            plt.ylabel("Distance [m]")
            plt.legend()

            latex_plot(self.t, self.bosanquet_modified, "bosanquet_modified")

if __name__ == "__main__":
    if True: # Question 1
        cap = Capillary()
        plt.figure()
        cap.normal_noslip_velocityprofile(plot=True)
        plt.show()

    if True: # Question 2
        cap = Capillary()
        plt.figure()
        cap.normal_noslip_timehistory(plot=True)
        cap.Lucas_washburn(plot=True)
        cap.Bosanquet(plot=True)
        plt.show()

    if True: # Question 3
        cap = Capillary()
        plt.figure()
        cap.normal_noslip_timehistory(plot=True)
        cap.normal_slip_timehistory(plot=True)
        cap.fillet_slip_timehistory(plot=True)  
        cap.fillet_noslip_timehistory(plot=True)
        plt.show()

    if True: # Question 5
        cap = Capillary()
        plt.figure()
        cap.fillet_noslip_closed_comp_timehistory(plot=True)
        cap.fillet_noslip_closed_uncomp_timehistory(plot=True)
        
        plt.figure()
        cap.fillet_noslip_closed_comp_velocityprofile(plot=True)
        cap.fillet_noslip_closed_uncomp_velocityprofile(plot=True)
        plt.show()



    if True: # Question 6
        plt.figure()
        cap = Capillary()
        cap.normal_noslip_timehistory(plot=True)
        cap.normal_slip_timehistory(plot=True)
        cap.Lucas_washburn(plot=True)
        cap.Bosanquet(plot=True)
        cap.Lucas_washburn_modified(plot=True)
        plt.show()

    if True: # Question 7
        cap = Capillary()
        plt.figure()
        cap.waterwater_noslip_timehistory(plot=True)
        cap.waterwater_slip_timehistory(plot=True)
        plt.show()

    if True: # Question 8
        cap = Capillary()
        plt.figure()
        cap.waterwater_noslip_timehistory(plot=True)
        cap.waterwater_slip_timehistory(plot=True)
        cap.Bosanquet_modified(plot=True)
        plt.show()
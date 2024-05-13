import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

class imbibition_simulation():
    def __init__(self, rho_l, rho_g, mu_l, mu_g, theta, gamma, H, L) -> None:
        self.rho_l = rho_l
        self.rho_g = rho_g
        self.mu_l = mu_l
        self.mu_g = mu_g
        self.theta = theta
        self.gamma = gamma
        self.h = H/2
        self.H = H
        self.L = L

        self.A_I = np.sqrt(2*self.gamma*np.cos(self.theta)/(self.rho_l*self.H))
        self.B = 12*self.mu_l/(self.rho_l*self.H**2)
        self.l_MB = None

        self.Alpha = mu_g*L/(mu_l*self.h)
        self.tC = gamma*np.cos(theta)/(6*mu_l*self.h)

        self._save_path = r"Jacob\Project3\Python_figures"

    def Lucas_Washburn(self, t):
        # Lucas-Washburn equation
        return np.sqrt(self.H*self.gamma*np.cos(self.theta)/(3*self.mu_l)*t)
    
    def Bosanquet(self, t):
        # Bosanquet equation
        return np.sqrt(2*self.A_I**2/self.B*(t-1/self.B*(1-np.exp(-self.B*t))))

    def get_lambda_modified_Bosanquet(self):

        # Define the variables
        L, B, A, t = sp.symbols('L B A t')

        # Define the function
        l = sp.Function('l')
        eq1 = sp.Eq(sp.Derivative(L * sp.Derivative(l(t),t), t) + B * L * sp.Derivative(l(t),t), A**2)
        ics = {l(0): 0, sp.Derivative(l(t),t).subs(t,0): 0}

        sol = sp.dsolve(eq1, ics=ics)

        l_func = sp.Lambda((t, A, B, L), sol.rhs)
        return l_func

    def modified_Bosanquet(self, t):
        if self.l_MB is None:
            self.l_MB = self.get_lambda_modified_Bosanquet()
        return [self.l_MB(_t, self.A_I, self.B, self.L) for _t in t]
        
    def modified_Lucas_Washburn(self, t):
        return self.h * (np.sqrt(t*self.tC + self.Alpha**2) - self.Alpha)

    def l_vs_t(self):
        data = r"Jacob\Project3\position_time_no_fillet.csv"
        df = pd.read_csv(data)
        t_data = df['Time'].values
        l_data = df['Position'].values
        t = np.linspace(0, 0.02, 1000)
        l_LW = self.Lucas_Washburn(t)
        l_B = self.Bosanquet(t)
        plt.figure(figsize=(7, 4))
        plt.plot(t, l_LW, label='Lucas-Washburn')
        plt.plot(t, l_B, label='Bosanquet')
        plt.plot(t_data, l_data, label='Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)  # Adjust linewidth for major grid lines
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Adjust linewidth for minor grid lines
        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t.pdf")

    def l_vs_t_modified_Lucas_Washburn(self):
        data = r"Jacob\Project3\position_time_no_fillet.csv"
        df = pd.read_csv(data)
        t_data = df['Time'].values
        l_data = df['Position'].values
        t = np.linspace(0, 0.02, 1000)
        l_B = self.Bosanquet(t)
        l_LW = self.Lucas_Washburn(t)
        l_MLW = self.modified_Lucas_Washburn(t)

        plt.figure(figsize=(7, 4))
        plt.plot(t, l_LW, label='Lucas-Washburn')
        plt.plot(t, l_B, label='Bosanquet')
        plt.plot(t_data, l_data, label='Simulation')
        plt.plot(t, l_MLW, label='Modified Lucas-Washburn')
        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)  # Adjust linewidth for major grid lines
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Adjust linewidth for minor grid lines
        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t_modified_Lucas_Washburn.pdf")

    def vel_profile_wo_fillet(self):
        data = r"Jacob\Project3\velocity_profile_no_fillet.csv"
        df = pd.read_csv(data)
        y = df['Centroid'].values / 1000
        u = df['Velocity'].values
        plt.plot(u, y, "o-")
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Z-position (m)')
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)  # Adjust linewidth for major grid lines
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)  # Adjust linewidth for minor grid lines
        plt.tight_layout()
        plt.savefig(self._save_path + r"\vel_profile_wo_fillet.pdf")

    def velocity_profile_no_slip_and_slip(self):
        data_no_slip = r"Jacob\Project3\velocity_profile_no_fillet.csv"
        data_slip = r"Jacob\Project3\velocity_profile_no_fillet_slip.csv"
        data_no_slip_fillet = r"Jacob\Project3\velocity_profile_fillet.csv"
        data_slip_fillet = r"Jacob\Project3\velocity_profile_fillet_slip.csv"
        df_no_slip = pd.read_csv(data_no_slip)
        df_slip = pd.read_csv(data_slip)
        df_no_slip_fillet = pd.read_csv(data_no_slip_fillet)
        df_slip_fillet = pd.read_csv(data_slip_fillet)
        y_no_slip = df_no_slip['Centroid'].values / 1000
        u_no_slip = df_no_slip['Velocity'].values
        y_slip = df_slip['Centroid'].values / 1000
        u_slip = df_slip['Velocity'].values
        y_no_slip_fillet = df_no_slip_fillet['Centroid'].values
        u_no_slip_fillet = df_no_slip_fillet['Velocity'].values
        y_slip_fillet = df_slip_fillet['Centroid'].values
        u_slip_fillet = df_slip_fillet['Velocity'].values
        plt.figure(figsize=(6, 4))
        plt.plot(u_no_slip, y_no_slip, "o-", label='No-slip', markersize=3)
        plt.plot(u_slip, y_slip, "o-", label='Slip', markersize=3)
        plt.plot(u_no_slip_fillet, y_no_slip_fillet, "o-", label='No-slip with fillet', markersize=3)
        plt.plot(u_slip_fillet, y_slip_fillet, "o-", label='Slip with fillet', markersize=3)
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Z-position (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self._save_path + r"\vel_profile_no_slip_and_slip.pdf")

    def l_vs_t_no_slip_and_slip(self):
        data_no_slip = r"Jacob\Project3\position_time_no_fillet.csv"
        data_slip = r"Jacob\Project3\position_time_no_fillet_slip.csv"
        data_no_slip_fillet = r"Jacob\Project3\position_time_fillet.csv"
        data_slip_fillet = r"Jacob\Project3\position_time_fillet_slip.csv"
        df_no_slip = pd.read_csv(data_no_slip)
        df_slip = pd.read_csv(data_slip)
        df_no_slip_fillet = pd.read_csv(data_no_slip_fillet)
        df_slip_fillet = pd.read_csv(data_slip_fillet)
        t_no_slip = df_no_slip['Time'].values
        l_no_slip = df_no_slip['Position'].values
        t_slip = df_slip['Time'].values
        l_slip = df_slip['Position'].values
        t_no_slip_fillet = df_no_slip_fillet['Time'].values
        l_no_slip_fillet = df_no_slip_fillet['Position'].values
        t_slip_fillet = df_slip_fillet['Time'].values
        l_slip_fillet = df_slip_fillet['Position'].values
        plt.figure(figsize=(6, 4))
        plt.plot(t_no_slip, l_no_slip, label='No-slip')
        plt.plot(t_slip, l_slip, label='Slip')
        plt.plot(t_no_slip_fillet, l_no_slip_fillet, label='No-slip with fillet')
        plt.plot(t_slip_fillet, l_slip_fillet, label='Slip with fillet')
        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t_no_slip_and_slip.pdf")

    def l_vs_t_closed(self):
        data_incompressible = r"Jacob\Project3\position_time_closed.csv"
        data_ideal= r"Jacob\Project3\position_time_closed_ideal.csv"
        df_incompressible = pd.read_csv(data_incompressible)
        df_ideal = pd.read_csv(data_ideal)
        t_incompressible = df_incompressible['Time'].values
        l_incompressible = df_incompressible['Position'].values
        t_ideal = df_ideal['Time'].values
        l_ideal = df_ideal['Position'].values
        plt.figure(figsize=(7, 4))
        plt.plot(t_incompressible, l_incompressible, label='Incompressible gas')
        plt.plot(t_ideal, l_ideal, label='Ideal gas')
        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t_closed.pdf")

    def l_vs_t_heavy_fluid(self):
        data_air = r"Jacob\Project3\position_time_no_fillet.csv"
        data_1_1 = r"Jacob\Project3\position_time_1_1.csv"
        data_1_2 = r"Jacob\Project3\position_time_1_2.csv"
        data_1_4 = r"Jacob\Project3\position_time_1_4.csv"
        data_1_8 = r"Jacob\Project3\position_time_1_8.csv"
        data_1_16 = r"Jacob\Project3\position_time_1_16.csv"

        df_air = pd.read_csv(data_air)
        df_1_1 = pd.read_csv(data_1_1)
        df_1_2 = pd.read_csv(data_1_2)
        df_1_4 = pd.read_csv(data_1_4)
        df_1_8 = pd.read_csv(data_1_8)
        df_1_16 = pd.read_csv(data_1_16)

        t_air = df_air['Time'].values
        l_air = df_air['Position'].values

        t_1_1 = df_1_1['Time'].values
        l_1_1 = df_1_1['Position'].values

        t_1_2 = df_1_2['Time'].values
        l_1_2 = df_1_2['Position'].values

        t_1_4 = df_1_4['Time'].values
        l_1_4 = df_1_4['Position'].values

        t_1_8 = df_1_8['Time'].values
        l_1_8 = df_1_8['Position'].values

        t_1_16 = df_1_16['Time'].values
        l_1_16 = df_1_16['Position'].values

        plt.figure(figsize=(8, 5))
        plt.plot(t_1_1, l_1_1, label=r'$\rho_g = 997 kg/m^3, \mu_g = 8.88\times 10^{-4}, Pa s$')
        plt.plot(t_1_2, l_1_2, label=r'$\rho_g = 498.5 kg/m^3, \mu_g = 4.44\times 10^{-4}, Pa s$')
        plt.plot(t_1_4, l_1_4, label=r'$\rho_g = 249.25 kg/m^3, \mu_g = 2.22\times 10^{-4}, Pa s$')
        plt.plot(t_1_8, l_1_8, label=r'$\rho_g = 124.625 kg/m^3, \mu_g = 1.11\times 10^{-4}, Pa s$')
        plt.plot(t_1_16, l_1_16, label=r'$\rho_g = 62.3125 kg/m^3, \mu_g = 5.55\times 10^{-5}, Pa s$')
        plt.plot(t_air, l_air, label='Air')

        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)

        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t_heavy_fluid.pdf")

    def l_vs_t_water_water(self):
        data_water = r"Jacob\Project3\position_time_1_1.csv"
        df_water = pd.read_csv(data_water)
        t_water = df_water['Time'].values
        l_water = df_water['Position'].values

        # Compare with the modified Bosanquet equation
        t = np.linspace(0, np.max(t_water), 1000)
        l_MB = self.modified_Bosanquet(t)
        L_B = self.Bosanquet(t)

        plt.figure(figsize=(7, 4))
        plt.plot(t, L_B, label='Bosanquet')
        plt.plot(t, l_MB, label='Modified Bosanquet')
        plt.plot(t_water, l_water, label='Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('Position of meniscus (m)')
        plt.legend()
        plt.grid(True, which='both', linestyle='-', linewidth=1.5)
        plt.minorticks_on()
        plt.grid(True, which='minor', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(self._save_path + r"\l_vs_t_water_water.pdf")

if __name__ == '__main__':

    plt.rc('legend', fontsize=10) # legend fontsize
    plt.rc('axes', labelsize=13)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=11)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=11)  # fontsize of the tick labels


    rho_l = 997.561
    rho_g = 1.18451
    mu_l = 8.8871e-4
    mu_g = 1.85508e-5
    theta = np.deg2rad(20)
    gamma = 0.072
    H = 1e-3
    L = 10e-3
    sim = imbibition_simulation(rho_l, rho_g, mu_l, mu_g, theta, gamma, H, L)

    sim.l_vs_t()
    # sim.vel_profile_wo_fillet()
    # sim.velocity_profile_no_slip_and_slip()
    # sim.l_vs_t_no_slip_and_slip()
    # sim.l_vs_t_closed()
    # sim.l_vs_t_heavy_fluid()
    # sim.l_vs_t_water_water()
    # sim.l_vs_t_modified_Lucas_Washburn()

    plt.show()
    

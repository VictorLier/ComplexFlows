import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter
import sys, os

class shape():
    def __init__(self):
        self.g = 9.82
        self.nu_c = 0.984*10**(-6)
        self.rho_c = 998
        self._s = None

        self.a = None
        self.b = None
        self.c = None
        self.V = None
        self.A = None
        self.d_D = None
        self.d_n = None
        self.d_p = None
        self.d_A = None
        self.P_p = None

        self.rho = None
        self.name = None
        self.drop: DropCalculations = None

        self._omega_s = None
        self._Re_p = None
        self._Re_n = None
        self._Re_D = None

        self._C_D = None
        self._C_n = None
        self._C_p = None

    def _initialize():
        raise NotImplementedError("This method must be defined in the subclass.")
    
    @property
    def omega_s(self):
        if self._omega_s is None:
            self._omega_s = np.abs(self.drop.get_mean_velocities()[2])/1000
        return self._omega_s
    
    @property
    def Re_p(self):
        if self._Re_p is None:
            self._Re_p = self.omega_s*self.d_p/self.nu_c
        return self._Re_p
    
    @property
    def Re_n(self):
        if self._Re_n is None:
            self._Re_n = self.omega_s*self.d_n/self.nu_c
        return self._Re_n
    
    @property
    def Re_D(self):
        if self._Re_D is None:
            self._Re_D = self.omega_s*self.d_D/self.nu_c
        return self._Re_D
    
    @property
    def s(self):
        if self._s is None:
            self._s = self.rho/self.rho_c
        return self._s

    @property
    def C_D(self):
        if self._C_D is None:
            self._C_D = 4*self.g*(self.s-1)*self.d_D/(3*self.omega_s**2)
        return self._C_D
    
    @property
    def C_n(self):
        if self._C_n is None:
            self._C_n = 4*self.g*(self.s-1)*self.d_n/(3*self.omega_s**2)
        return self._C_n

    @property
    def C_p(self):
        if self._C_p is None:
            self._C_p = 4*self.g*(self.s-1)*self.d_p/(3*self.omega_s**2)
        return self._C_p

    def plot_Re_vs_C(self):
        # Plot Re_n vs C_n and Re_D vs C_D in two suplots. Add mean line of the mean values.
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        fig.suptitle('Reynolds number vs drag coefficients for ' + self.name)
        axs[0].plot(self.Re_n, self.C_n, 'o')
        # Mean line
        axs[0].axhline(y=np.mean(self.C_n), color='r', linestyle='-', label='Mean')
        axs[0].set_xlabel('Reynolds number')
        axs[0].set_ylabel('Drag coefficient')
        axs[0].set_title('Re_n vs C_n')

        axs[1].plot(self.Re_D, self.C_D, 'o')
        # Mean line
        axs[1].axhline(y=np.mean(self.C_D), color='r', linestyle='-', label='Mean')
        axs[1].set_xlabel('Reynolds number')
        axs[1].set_ylabel('Drag coefficient')
        axs[1].set_title('Re_D vs C_d')

        axs[2].plot(self.Re_p, self.C_p, 'o')
        # Mean line
        axs[2].axhline(y=np.mean(self.C_p), color='r', linestyle='-', label='Mean')
        axs[2].set_xlabel('Reynolds number')
        axs[2].set_ylabel('Drag coefficient')
        axs[2].set_title('Re_p vs C_p')
        


        plt.tight_layout()

    def plot_termal_velocity(self):
        # Plot the termal velocity vs track number
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle('Termal velocity vs track number for ' + self.name)
        ax.plot(np.arange(len(self.drop.drops)), self.drop.get_mean_velocities()[2], 'o')
        ax.set_xlabel('Track number')
        ax.set_ylabel('Termal velocity [mm/s]')
        ax.set_title('Termal velocity vs track number')

class SingleDropCalculations():
    def __init__(self, file_path, shape_name, track_number):
        self._file_path = file_path
        self._shape_name = shape_name
        self._track_number = track_number
        self._sample_rate = 1/100
        self.do_calculations()

    def do_calculations(self, plot = False):
        column_names = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z']
        file_path = os.path.join(os.getcwd(), r"Jacob\Project1\41321_hand-in_data_v2")
        track_path = os.path.join(file_path, self._shape_name + r'\Tracks\track' + str(self._track_number).zfill(2) + '.csv')
        track = pd.read_csv(track_path, comment='#', names=column_names)

        # Changing from degrees to radians.
        track[['ori_x', 'ori_y', 'ori_z']] = track[['ori_x', 'ori_y', 'ori_z']].apply(lambda x : np.deg2rad(x), axis=0)

        # If you would like to filter the signal you could among many things do as below
        # Experiment with the window_length adn polyorder.
        self.ori_x_filtered = savgol_filter(track.ori_x, window_length=10, polyorder=1)
        self.ori_y_filtered = savgol_filter(track.ori_y, window_length=10, polyorder=1)
        self.ori_z_filtered = savgol_filter(track.ori_z, window_length=10, polyorder=1)
        
        # Now I will show how to interpret this weird rodriguez rotation vector.
        
        # 1.
        # A CAD file consists of a series of triangles. Therefore for a 
        # cuboid each rectangular face consists of 2 triangles and thus in total 12 triangles.
        # Each triangle is defined from 3 points in 3D space. Thus we get 12 triangles each
        # consisting of 3 points with 3 coordinates.
        # For simplicity I have added this as a .npy file which is loaded in the lines below.
        
        shape_path = os.path.join(file_path, self._shape_name + '/' + self._shape_name + '.npy')
        shape_file = np.load(shape_path)
        vectors = shape_file - np.mean(shape_file.reshape(-1,3))  # Center shape in case 

        # 1. Here we illustrate in 3D the cube as it moves and as it rotates.
        positions = track[['pos_x', 'pos_y', 'pos_z']].to_numpy()
        rodriguez_vectors = track[['ori_x', 'ori_y', 'ori_z']].to_numpy()
        plot_every_X_instances = 25
        vectors_rotated_and_translated = []     
        for i in range(0, len(track), plot_every_X_instances):
            # First we rotate the shape
            rodriguez_vector = rodriguez_vectors[i]
            # Get rotation matrix and apply to vectors
            rot_matrix = self.get_rotation_matrix_from_rotvec( np.rad2deg(rodriguez_vector) )
            rotated_vectors = vectors.reshape(-1,3) @ rot_matrix.transpose()
            rotated_vectors = rotated_vectors.reshape(-1,3,3)
            
            # After rotation we move all vectors
            translation_vector = positions[i]
            vectors_rotated_and_translated.append( rotated_vectors + translation_vector )

        # Finding velocities
        # To get velocity from position we need to take the derivate of position. 
        # This can be done numerically using the np.gradient function. 
        # For reference check the documentation https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        self.velocities = np.gradient(positions, self._sample_rate, axis=0)
        """
            NOTE: At some point the particle stop being tracked by one camera set starts being 
            tracked by the next set of cameras. This gives a large spike in velocity.
            Clearly this is not physical so you should deal with this. The simplest option is 
            to remove this point from your measurements.
        """
        # Finding relative rotation.
        """ 
            In this code the rotation is the rotation from the CAD orientation of the particle
            to the rotation we measure in the experiment. If we want to find the rotation
            from one state (time-step) to another we can do the following
        """
        
        relative_to_first = []
        for idx, vec in enumerate(rodriguez_vectors, start = 1):
            relative_to_first.append(self.relative_rotation_vector(rodriguez_vectors[0], vec) )
        
        self.relative_to_first = np.array(relative_to_first)

        # Getting x, y, z components of the velocity
        # self.vel_x = self.velocities[:,0]
        # self.vel_y = self.velocities[:,1]
        # self.vel_z = self.velocities[:,2]

        self.vel_x = self.velocities[:,0]
        self.vel_y = self.velocities[:,1]
        self.vel_z = self.velocities[:,2]

        # Find values over 3 standard deviations from the mean
        std_x = np.std(self.vel_x)
        std_y = np.std(self.vel_y)
        std_z = np.std(self.vel_z)

        # Change values over 3 standard deviations from the mean to the value before
        idx_x = np.where(np.abs(self.vel_x - np.mean(self.vel_x)) > 3*std_x)
        idx_y = np.where(np.abs(self.vel_y - np.mean(self.vel_y)) > 3*std_y)
        idx_z = np.where(np.abs(self.vel_z - np.mean(self.vel_z)) > 3*std_z)
        self.vel_x_filtered = self.vel_x
        self.vel_y_filtered = self.vel_y
        self.vel_z_filtered = self.vel_z
        for idx in idx_x:
            try:
                # self.vel_x[idx] = np.mean(self.vel_x[idx-5:idx+6])
                self.vel_x_filtered[idx] = (self.vel_x_filtered[idx-3] + self.vel_x_filtered[idx+3])/2
            except:
                pass
        for idx in idx_y:
            try:
                # self.vel_y[idx] = np.mean(self.vel_y[idx-5:idx+6])
                self.vel_y_filtered[idx] = (self.vel_y_filtered[idx-3] + self.vel_y_filtered[idx+3])/2
            except:
                pass
        for idx in idx_z:
            try:
                # self.vel_z[idx] = np.mean(self.vel_z[idx-5:idx+6])
                self.vel_z_filtered[idx] = (self.vel_z_filtered[idx-3] + self.vel_z_filtered[idx+3])/2
            except:
                pass

        # Smooth the z velocity
        self.vel_x_filtered = savgol_filter(self.vel_x_filtered, window_length=10, polyorder=1)
        self.vel_y_filtered = savgol_filter(self.vel_y_filtered, window_length=10, polyorder=1)
        self.vel_z_filtered = savgol_filter(self.vel_z_filtered, window_length=10, polyorder=1)

        if plot:
            # Plotting position
            self.plot_xyz(track.pos_x, track.pos_y, track.pos_z, ['Position x [mm]','Position y [mm]', 'Position z [mm]'], 'Position of Particle')

            # Plotting the velocity of the particle
            self.plot_xyz(self.vel_x, self.vel_y, self.vel_z, ['Velocity x [mm/s]', 'Velocity y [mm/s]', 'Velocity z [mm/s]'], 'Velocity of Particle')

            # Plotting the orientation of the particle
            self.plot_xyz(self.ori_x_filtered, self.ori_y_filtered, self.ori_z_filtered, ['Orientation x [rad]', 'Orientation y [rad]', 'Orientation z [rad]'], 'Filtered Orientation of Particle')

            # Plotting the orientation of the particle
            self.plot_orientation(track, vectors_rotated_and_translated)

            # Plotting the orientation of the particle relative to the first appearance
            self.plot_xyz(self.relative_to_first[:,0], self.relative_to_first[:,1], self.relative_to_first[:,2], ['Orientation x [rad]',  'Orientation y [rad]',  'Orientation z [rad]'], 'Relative to first appearance orientation')

    def _gauss_interpolate(self, x):
        """ Return fractional peak position assuming Guassian shape
            x is a numpy vector with 3 elements,
            ifrac returned is relative to center element.
        """
        assert (x[1] >= x[0]) and (x[1] >= x[2]), 'Peak must be at center element'
        # avoid log(0) or divide 0 error
        if all(x > 0):
            r = np.log(x)
            ifrac = (r[0] - r[2]) / (2 * r[0] - 4 * r[1] + 2 * r[2])
        else:
            # print("using centroid in gauss_interpolate")
            ifrac = (x[2] - x[0]) / sum(x)
        return ifrac

    def spectral_analysis(self):
        #self._nw = 1024
        self._nw = len(self.vel_x)
        # for vel in [self.vel_x, self.vel_y, self.vel_z]:
        for vel in [self.vel_z]:
            f = np.arange(self._nw) / (self._nw * self._sample_rate)
            window = np.zeros(self._nw)
            window[0:len(vel)] = vel
            sigf = np.fft.fft(window)
            sigspec = np.real(sigf * sigf.conjugate()) / (self._nw*self._sample_rate)
            # ipeak = np.argmax(sigspec) # Old method: Did not work beacause of symmetric spectrum - sometimes the peak was chosen to be the wrong side of the spectrum
            ipeak = np.argmax(sigspec[0:self._nw//2])
            plt.figure()
            plt.loglog(f[0:self._nw//2], sigspec[0:self._nw//2], label = 'Spectrum') # OBS: Start at 1 to avoid log(0)
            #plt.plot(f[ipeak], sigspec[ipeak], 'ro', label = 'Peak')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [mm^2/s^2]')
            plt.title('Spectrum of velocity')
            plt.legend()
            plt.show()
            # ifrac = self._gauss_interpolate(sigspec[ipeak-1:ipeak+2])
            # freq = (ipeak + ifrac) / (self._nw * self._sample_rate)
            stop = True

    def plot_xyz(self, x, y, z, ylabels, title):
        # Plotting position
        fig, ax = plt.subplots(3, 1, figsize=(7,7))
        ax[0].plot( x )
        ax[0].set_ylabel( ylabels[0] )
        
        ax[1].plot( y )
        ax[1].set_ylabel( ylabels[1] )
        
        ax[2].plot( z )
        ax[2].set_ylabel( ylabels[2] )
        
        fig.suptitle( title )

        return fig, ax

    def get_rotation_matrix_from_rotvec(self, rotvec):
        """
        returns the rotation matrix based on a rodriguez rotation vector. But since everyone
        has seen rotation matrices I think this is the easiest method. First part of the code
        just ensures we dont divide by zero. Second part of the code is based on the following
        wikipedia page: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        If youre interested have a look under "Matrix Notation" but it does not matter. 
        The rodriguez vector is simply a way of storing a rotation. Nothing more to it.

        Parameters
        ----------
        rotvec : float64[3]
            rotvec: rotation vector. The magnitude of the vector is in degrees.

        Returns
        -------
        rot_matrix : float64[3,3]
            Rotation matrix.

        """
        # Norm of rotation vector
        norm = np.sqrt(rotvec[0]**2 + rotvec[1]**2 + rotvec[2]**2 )
        
        # Error check if 0 rotation is requested we want to avoid division by 0
        if np.isclose(norm, 0):
            return np.eye(3, dtype=np.float64)
        
        # convert from degrees to radians
        theta = np.deg2rad( norm )
        
        # Use Rodriguez formula to create rotation matrix
        k = rotvec / norm
        K = np.array([[0, -k[2],    k[1]],
                    [k[2],  0,   -k[0]],
                    [-k[1], k[0],  0]  ])
        rot_matrix = np.eye(3) + np.sin(theta) * K + (1-np.cos(theta)) * K @ K
        
        return rot_matrix

    def get_rodriguez_from_rotation_matrix(self, R):
        """Convert a rotation matrix to a Rodrigues vector."""    
        theta = np.arccos((np.trace(R) - 1) / 2)
        
        if theta < 1e-6:
            # The rotation is very small, close to zero
            return np.array([0, 0, 0])
        elif theta < np.pi:
            vector = np.array([R[2, 1] - R[1, 2],
                        R[0, 2] - R[2, 0],
                        R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
            return vector * theta
        else:
            # Handle the case where theta is around pi, where the above formula might be numerically unstable
            # Find the largest diagonal element and compute its corresponding Rodrigues vector component
            if R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
                r1 = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
                r2 = (R[0, 1] + R[1, 0]) / (4 * r1)
                r3 = (R[0, 2] + R[2, 0]) / (4 * r1)
            elif R[1, 1] > R[2, 2]:
                r2 = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2
                r1 = (R[0, 1] + R[1, 0]) / (4 * r2)
                r3 = (R[1, 2] + R[2, 1]) / (4 * r2)
            else:
                r3 = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2
                r1 = (R[0, 2] + R[2, 0]) / (4 * r3)
                r2 = (R[1, 2] + R[2, 1]) / (4 * r3)
            vector = np.array([r1, r2, r3])
            if theta > np.pi:  # Adjust sign if necessary
                vector = -vector
            return vector * theta

    def relative_rotation_vector(self, vec1, vec2):
            
            R1 = self.get_rotation_matrix_from_rotvec(np.rad2deg(vec1))
            R2 = self.get_rotation_matrix_from_rotvec(np.rad2deg(vec2))
            
            # Compute relative matrix
            R_relative = np.dot(R2, R1.T)
            
            # Convert back to vector
            relative_vec = self.get_rodriguez_from_rotation_matrix(R_relative)
            
            return relative_vec    

    def plot_orientation(self, track, vectors_rotated_and_translated):
        # First we create a plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # Note we are using projection='3d' since we want a 3D figure
        
        # Now we loop over every shape instance we can find
        for i in range(len(vectors_rotated_and_translated)):
            # Creating a polycollection for this set of triangles
            poly_collection = Poly3DCollection(vectors_rotated_and_translated[i], 
                                            alpha=1.0, 
                                            facecolors='gray', 
                                            edgecolors='black')
            # Adding the polycollection of triangles to the plot axes ax.
            ax.add_collection(poly_collection)
        # Fixing the aspect ratio and setting labels
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_zlabel('z [mm]')
        ax.set_zlim(track.pos_z.min(), track.pos_z.max())
        ax.set_xlim(-track.pos_z.min()/2, track.pos_z.min()/2)
        ax.set_ylim(-track.pos_z.min()/2, track.pos_z.min()/2)
        ax.set_aspect('equal')

        return fig, ax

class DropCalculations():
    def __init__(self, file_path, shape_name):
        self._file_path = file_path
        self._shape_name = shape_name
        self._sample_rate = 1/100
        self.do_calculations()
        self.N = self.min_time_steps()

        self.x_vel = None
        self.y_vel = None
        self.z_vel = None
        self.x_vel_filtered = None
        self.y_vel_filtered = None
        self.z_vel_filtered = None
        self.x_ori = None
        self.y_ori = None
        self.z_ori = None
        self.speed = None
        self.speed_filtered = None
    
    def do_calculations(self, plot = False):
        """
        For all tracks in the shape, do the calculations. Save objects in a list.
        """
        # Get all track files
        track_files = [f for f in os.listdir(os.path.join(self._file_path, self._shape_name, 'Tracks')) if f.endswith('.csv')]
        self.drops = []
        for track_file in track_files:
            track_number = int(track_file.split('track')[1].split('.')[0])
            drop = SingleDropCalculations(self._file_path, self._shape_name, track_number)
            self.drops.append(drop)
        
        stop = True

    def min_time_steps(self):
        """
        Find the track with the minimum time steps and return the number of time steps.
        """
        n_time_steps = np.inf
        for drop in self.drops:
            n_time_steps = min(n_time_steps, len(drop.vel_x))
        return int(n_time_steps)

    def get_velocities(self):
        # Get x, y, z components of the velocity such that a column is a track and a row is a time step
        self.x_vel = np.zeros((self.N, len(self.drops)))
        self.y_vel = np.zeros((self.N, len(self.drops)))
        self.z_vel = np.zeros((self.N, len(self.drops)))

        self.x_vel_filtered = np.zeros((self.N, len(self.drops)))
        self.y_vel_filtered = np.zeros((self.N, len(self.drops)))
        self.z_vel_filtered = np.zeros((self.N, len(self.drops)))

        for i, drop in enumerate(self.drops):
            self.x_vel[:,i] = drop.vel_x[:self.N]
            self.y_vel[:,i] = drop.vel_y[:self.N]
            self.z_vel[:,i] = drop.vel_z[:self.N]
            self.x_vel_filtered[:,i] = drop.vel_x_filtered[:self.N]
            self.y_vel_filtered[:,i] = drop.vel_y_filtered[:self.N]
            self.z_vel_filtered[:,i] = drop.vel_z_filtered[:self.N]

        return self.x_vel, self.y_vel, self.z_vel
    
    def get_orientations(self):
        # Get x, y, z components of the orientation such that a column is a track and a row is a time step
        self.x_ori = np.zeros((self.N, len(self.drops)))
        self.y_ori = np.zeros((self.N, len(self.drops)))
        self.z_ori = np.zeros((self.N, len(self.drops)))
        self.x_ori_rel_to_first = np.zeros((self.N, len(self.drops)))
        self.y_ori_rel_to_first = np.zeros((self.N, len(self.drops)))
        self.z_ori_rel_to_first = np.zeros((self.N, len(self.drops)))

        for i, drop in enumerate(self.drops):
            self.x_ori[:,i] = drop.ori_x_filtered[:self.N]
            self.y_ori[:,i] = drop.ori_y_filtered[:self.N]
            self.z_ori[:,i] = drop.ori_z_filtered[:self.N]
            self.x_ori_rel_to_first[:,i] = drop.relative_to_first[:self.N,0]
            self.y_ori_rel_to_first[:,i] = drop.relative_to_first[:self.N,1]
            self.z_ori_rel_to_first[:,i] = drop.relative_to_first[:self.N,2]

        return self.x_ori, self.y_ori, self.z_ori
    
    def _gauss_interpolate(self, x):
        """ Return fractional peak position assuming Guassian shape
            x is a numpy vector with 3 elements,
            ifrac returned is relative to center element.
        """
        assert (x[1] >= x[0]) and (x[1] >= x[2]), 'Peak must be at center element'
        # avoid log(0) or divide 0 error
        if all(x > 0):
            r = np.log(x)
            ifrac = (r[0] - r[2]) / (2 * r[0] - 4 * r[1] + 2 * r[2])
        else:
            # print("using centroid in gauss_interpolate")
            ifrac = (x[2] - x[0]) / sum(x)
        return ifrac

    def do_spectral_analysis_ori(self):
        if self.x_ori is None or self.y_ori is None or self.z_ori is None:
            self.get_orientations()

        for ori, ori_str in [(self.x_ori, "x orientation"), (self.y_ori, "y orientation"), (self.z_ori, "z orientation")]:
            f = np.arange(self.N) / (self.N * self._sample_rate)
            sigf = np.fft.fft(ori, axis=0)
            sigspec = np.real(sigf * sigf.conjugate()) / (self.N*self._sample_rate)
            sigmean = np.mean(sigspec[0:self.N//2], axis=1)
            ipeak = np.argmax(sigmean, axis=0)
            iremove = 1
            while True:
                # Incase the peak is located at the edge of the spectrum, find the second higest peak
                if sigmean[ipeak+1] > sigmean[ipeak] or sigmean[ipeak-1] > sigmean[ipeak] or ipeak == 0:
                    ipeak = np.argmax(sigmean[0:self.N//2][1*iremove:])
                    iremove += 1
                else:
                    break
            ifrac = self._gauss_interpolate(sigmean[max(0,ipeak-1): max(3, ipeak+2)])
            freq = (ipeak + ifrac) / (self.N * self._sample_rate)

            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle('Analysis ' + ori_str)
            # Plot Spectrum
            axs[0].set_title("Spectrum of {}. Frequency: {:.2f} Hz".format(ori_str, freq))
            axs[0].loglog(f[0:self.N//2], sigmean[0:self.N//2], label='Spectrum')
            axs[0].set_xlabel('Frequency [Hz]')
            axs[0].set_ylabel('Power [mm^2/s^2]')
            axs[0].legend()
            axs[0].grid()

            # Plot Velocity
            axs[1].plot(np.arange(self.N) * self._sample_rate, ori)
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Velocity [mm/s]')
            axs[1].set_title('Velocity of particle')
            axs[1].grid(which='both', axis='both')
            axs[1].minorticks_on()
            axs[1].grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')

            plt.tight_layout()

            stop = True
    
    def do_spectral_analysis(self):
        if self.x_vel is None or self.y_vel is None or self.z_vel is None or self.x_vel_filtered is None or self.y_vel_filtered is None or self.z_vel_filtered is None:
            self.get_velocities()

        for vel, vel_str in [(self.x_vel_filtered, "x velocity"), (self.y_vel_filtered, "y velocity"), (self.z_vel_filtered, "z velocity")]:
            f = np.arange(self.N) / (self.N * self._sample_rate)
            sigf = np.fft.fft(vel, axis=0)
            sigspec = np.real(sigf * sigf.conjugate()) / (self.N*self._sample_rate)
            sigmean = np.mean(sigspec[0:self.N//2], axis=1)
            ipeak = np.argmax(sigmean, axis=0)
            iremove = 1
            while True:
                # Incase the peak is located at the edge of the spectrum, find the second higest peak
                if sigmean[ipeak+1] > sigmean[ipeak] or sigmean[ipeak-1] > sigmean[ipeak] or ipeak == 0:
                    ipeak = np.argmax(sigmean[0:self.N//2][1*iremove:])
                    iremove += 1
                else:
                    break
            ifrac = self._gauss_interpolate(sigmean[max(0,ipeak-1): max(3, ipeak+2)])
            freq = (ipeak + ifrac) / (self.N * self._sample_rate)

            fig, axs = plt.subplots(2, 1, figsize=(10, 8))
            fig.suptitle('Analysis ' + vel_str)
            # Plot Spectrum
            axs[0].set_title("Spectrum of {}. Frequency: {:.2f} Hz".format(vel_str, freq))
            axs[0].loglog(f[0:self.N//2], sigmean[0:self.N//2], label='Spectrum')
            axs[0].set_xlabel('Frequency [Hz]')
            axs[0].set_ylabel('Power [mm^2/s^2]')
            axs[0].legend()
            axs[0].grid()

            # Plot Velocity
            axs[1].plot(np.arange(self.N) * self._sample_rate, vel)
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Velocity [mm/s]')
            axs[1].set_title('Velocity of particle')
            axs[1].grid(which='both', axis='both')
            axs[1].minorticks_on()
            axs[1].grid(which='minor', axis='both', linestyle=':', linewidth='0.5', color='black')

            plt.tight_layout()

            stop = True
    
    def get_mean_velocities(self):
        if self.x_vel is None or self.y_vel is None or self.z_vel is None or self.x_vel_filtered is None or self.y_vel_filtered is None or self.z_vel_filtered is None:
            self.get_velocities()

        x_vel_mean = np.zeros(len(self.drops))
        y_vel_mean = np.zeros(len(self.drops))
        z_vel_mean = np.zeros(len(self.drops))
        for i in range(len(self.drops)):
            std_x = np.std(self.x_vel[:,i])
            std_y = np.std(self.y_vel[:,i])
            std_z = np.std(self.z_vel[:,i])

            # Change values over 3 standard deviations from the mean to the value before
            valid_mask_x = np.abs(self.x_vel[:,i] - np.mean(self.x_vel[:,i])) < 3*std_x
            valid_mask_y = np.abs(self.y_vel[:,i] - np.mean(self.y_vel[:,i])) < 3*std_y
            valid_mask_z = np.abs(self.z_vel[:,i] - np.mean(self.z_vel[:,i])) < 3*std_z

            # Take the mean of the values
            x_vel_mean[i] = np.mean(self.x_vel[valid_mask_x, i])
            y_vel_mean[i] = np.mean(self.y_vel[valid_mask_y, i])
            z_vel_mean[i] = np.mean(self.z_vel[valid_mask_z, i])

        return x_vel_mean, y_vel_mean, z_vel_mean

    def plot_mean_velocities(self):
        x_vel_mean, y_vel_mean, z_vel_mean = self.get_mean_velocities()
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle('Mean velocity vs track number for ' + self._shape_name)
        ax.plot(np.arange(len(self.drops)), z_vel_mean, 'o')
        ax.set_xlabel('Track number')
        ax.set_ylabel('Mean velocity [mm/s]')
        ax.set_title('Mean velocity vs track number')

    def get_speed(self):
        if self.x_vel is None or self.y_vel is None or self.z_vel is None or self.x_vel_filtered is None or self.y_vel_filtered is None or self.z_vel_filtered is None:
            self.get_velocities()
        
        # Get the total speed
        self.speed = np.sqrt(self.x_vel**2 + self.y_vel**2 + self.z_vel**2)
        self.speed_filtered = np.sqrt(self.x_vel_filtered**2 + self.y_vel_filtered**2 + self.z_vel_filtered**2)

    def plot_speed_vs_orientation(self):
        if self.speed is None or self.speed_filtered is None:
            self.get_speed()
        
        if self.x_ori is None or self.y_ori is None or self.z_ori is None:
            self.get_orientations()

        # # Plot the speed vs the orientation, in the x, y and z direction and the speed as color.
        # # Do it as a 3D plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(len(self.drops)):
        #     ax.scatter(self.x_ori_rel_to_first[:,i], self.y_ori_rel_to_first[:,i], np.abs(self.z_vel_filtered[:,i]))
        # ax.set_xlabel('X orientation')
        # ax.set_ylabel('Y orientation')
        # ax.set_zlabel('Z velocity [mm/s]')
        # ax.set_title('Speed vs orientation')
            

        # Plot the speed vs the orientation
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        fig.suptitle('Speed vs orientation for ' + self._shape_name)
        for i in range(len(self.drops)):
            ax.scatter(np.abs(self.y_ori[:,i] - np.mean(self.y_ori[:,i])), np.abs(self.z_vel_filtered[:,i]))
            # ax.scatter(np.abs(self.y_ori[:,i] - np.mean(self.y_ori[:,i]))[1:], np.diff(np.abs(self.z_vel_filtered[:,i])))
        ax.set_xlabel('Y orientation')
        ax.set_ylabel('Z velocity [mm/s]')
        ax.set_title('Speed vs orientation')
        stop = True

class cube(shape):
    def __init__(self, a, rho, file_path, name):
        super().__init__()
        self._initialize(a, rho)
        self.name = name
        self.drop = DropCalculations(file_path, name)
        
    def _initialize(self, a, rho):
        self.a = a
        self.b = a
        self.c = a
        self.V = self.a**3
        self.A = self.a**2
        self.d_D = self.a*3/2
        self.d_n = self.a*(6/np.pi)**(1/3)
        self.d_p = self.a
        self.d_A = np.sqrt(4*self.A/np.pi)
        self.P_p = self.a*4

        self.rho = rho

class square_prism(shape):
    def __init__(self, a, b, rho, file_path, name):
        super().__init__()
        self._initialize(a, b, rho)
        self.name = name
        self.drop = DropCalculations(file_path, name)
        
    def _initialize(self, a, b, rho):
        self.a = a
        self.b = b
        self.c = b
        self.V = self.a*self.b*self.b
        self.A = self.a*self.b
        self.d_D = self.b*3/2
        self.d_n = (6/np.pi*self.a*self.b*self.b)**(1/3)
        self.d_p = np.sqrt(self.a*self.b)
        self.d_A = np.sqrt(4*self.A/np.pi)
        self.P_p = 2*self.a + 2*self.b

        self.rho = rho

class rectangular_prism(shape):
    def __init__(self, a, b, c, rho, file_path, name):
        super().__init__()
        self._initialize(a, b, c, rho)
        self.name = name
        self.drop = DropCalculations(file_path, name)
        
    def _initialize(self, a, b, c, rho):
        self.a = a
        self.b = b
        self.c = c
        self.V = self.a*self.b*self.c
        self.A = self.a*self.b
        self.d_D = self.c*3/2
        self.d_n = (6/np.pi*self.a*self.b*self.c)**(1/3)
        self.d_p = np.sqrt(self.a*self.b)
        self.d_A = np.sqrt(4*self.A/np.pi)
        self.P_p = 2*self.a + 2*self.b

        self.rho = rho        

if __name__ == '__main__':   
    # Path to the data
    file_path = os.path.join(os.getcwd(), r"Jacob\Project1\41321_hand-in_data_v2")
    shape_name = "Rectangle_2.5_10_5"
    # shape_name = "Cube_5mm"
    # shape_name = "4x4x8"

    if True:
        track_number = 1
        single_drop = SingleDropCalculations(file_path, shape_name, track_number)
        single_drop.do_calculations(plot=True)

    if False:
        test_drops = DropCalculations(file_path, shape_name)
        test_drops.do_spectral_analysis()
        test_drops.do_spectral_analysis_ori()
        

    if True:
        test_drops = DropCalculations(file_path, shape_name)
        test_drops.get_mean_velocities()
        test_drops.plot_speed_vs_orientation()

    if False:
        rho = 1030
        cube_test = cube(5e-3, rho, file_path, "Cube_5mm")
        square_test = square_prism(4e-3, 8e-3, rho, file_path, "4x4x8")
        rectangular_test = rectangular_prism(2.5e-3, 5e-3, 10e-3, rho, file_path, "Rectangle_2.5_10_5")

        cube_test.plot_Re_vs_C()
        square_test.plot_Re_vs_C()
        rectangular_test.plot_Re_vs_C()

        stop = True





    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
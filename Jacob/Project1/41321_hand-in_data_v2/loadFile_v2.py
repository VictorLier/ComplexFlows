import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.signal import savgol_filter
import sys, os

class DropCalculations():
    def __init__(self, file_path, shape_name, track_number):
        self._file_path = file_path
        self._shape_name = shape_name
        self._track_number = track_number
        self._sample_rate = 1/100
        self.do_calculations()

        
    
    def do_calculations(self, plot = False):
        column_names = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z']
        file_path = os.path.join(os.getcwd(), r"Jacob\Project1\41321_hand-in_data_v2")
        track_path = os.path.join(file_path, shape_name + r'\Tracks\track' + str(track_number).zfill(2) + '.csv')
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
        
        shape_path = os.path.join(file_path, shape_name + '/' + shape_name + '.npy')
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
        
        relative_to_first = np.array(relative_to_first)

        # Getting x, y, z components of the velocity
        self.vel_x = self.velocities[:,0]
        self.vel_y = self.velocities[:,1]
        self.vel_z = self.velocities[:,2]

        if plot:
            # Plotting position
            self.plot_xyz(track.pos_x, track.pos_y, track.pos_z, ['Position x [mm]','Position y [mm]', 'Position z [mm]'], 'Position of Particle')

            # Plotting the velocity of the particle
            self.plot_xyz(self.vel_x, self.vel_y, self.vel_z, ['Velocity x [mm/s]', 'Velocity y [mm/s]', 'Velocity z [mm/s]'], 'Velocity of Particle')

            # # Plotting the orientation of the particle
            # self.plot_xyz(ori_x_filtered, ori_y_filtered, ori_z_filtered, ['Orientation x [rad]', 'Orientation y [rad]', 'Orientation z [rad]'], 'Filtered Orientation of Particle')

            # # Plotting the orientation of the particle
            # self.plot_orientation(vectors_rotated_and_translated)

            # # Plotting the orientation of the particle relative to the first appearance
            # self.plot_xyz( relative_to_first[:,0], relative_to_first[:,1], relative_to_first[:,2], ['Orientation x [rad]',  'Orientation y [rad]',  'Orientation z [rad]'], 'Relative to first appearance orientation')

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
        for vel in [self.vel_x, self.vel_y, self.vel_z]:
            f = np.arange(self._nw) / (self._nw * self._sample_rate)
            window = np.zeros(self._nw)
            window[0:len(vel)] = vel
            sigf = np.fft.fft(window)
            sigspec = np.real(sigf * sigf.conjugate()) / (self._nw*self._sample_rate)
            # ipeak = np.argmax(sigspec) # Old method: Did not work beacause of symmetric spectrum - sometimes the peak was chosen to be the wrong side of the spectrum
            ipeak = np.argmax(sigspec[0:self._nw//2])
            plt.figure()
            plt.plot(f[0:self._nw//2], sigspec[0:self._nw//2], label = 'Spectrum')
            #plt.plot(f[ipeak], sigspec[ipeak], 'ro', label = 'Peak')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power [mm^2/s^2]')
            plt.title('Spectrum of velocity')
            plt.legend()
            plt.show()
            ifrac = self._gauss_interpolate(sigspec[ipeak-1:ipeak+2])
            freq = (ipeak + ifrac) / (self._nw * self._sample_rate)
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

    def plot_orientation(self, vectors_rotated_and_translated):
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

if __name__ == '__main__':   
    # Path to the data
    file_path = os.path.join(os.getcwd(), r"Jacob\Project1\41321_hand-in_data_v2")
    shape_name = "Rectangle_2.5_10_5"
    track_number = 1
    drop = DropCalculations(file_path, shape_name, track_number)

    drop.spectral_analysis()
    
   
    # nw = 1024
    # f = np.arange(nw) / (self._nw * self._dt)
    # #ipick = int(np.round(0.386837436/ self._dt)) # select a burst spectrum to plot
    # if len(istart) > 0:
    #     ipick = istart[0]
    # else:
    #     ipick = None

    # for i, j in zip(istart, iend):
    #     if j-i > self._nw:
    #         continue  # burst too long to handle
    #     window = np.zeros(self._nw)
    #     window[0:j-i] = sigfilt[i:j]
    #     sigf = np.fft.fft(window)
    #     sigspec = np.real(sigf * sigf.conjugate()) / (self._nw*self._dt)
    #     # ipeak = np.argmax(sigspec) # Old method: Did not work beacause of symmetric spectrum - sometimes the peak was chosen to be the wrong side of the spectrum
    #     ipeak = np.argmax(sigspec[0:self._nw//2])





    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
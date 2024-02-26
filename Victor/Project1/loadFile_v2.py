
"""
    README!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    This introductory code was written somewhat like a jupyter notebook. So take it one 
    section at a time and try to understand what is going on. I tried to add descriptions 
    but if something is unclear please send me an email at "seeb@dtu.dk" and I will fix it.
    
    While there are many lines of code here which can seem daunting notice that most of 
    it is plotting boiler plate like setting limits and labels on axes.




"""




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

plt.close('all')


def plot_xyz(x, y, z, ylabels, title):
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


def get_rotation_matrix_from_rotvec(rotvec):
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


def get_rodriguez_from_rotation_matrix(R):
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


def remove_outliers_iqr(data, percentile=0.25):
  """
  Removes outliers from a NumPy array using the Interquartile Range (IQR) method.

  Args:
      data: The NumPy array containing the data.
      percentile: The percentile used to define the IQR (default: 0.25).

  Returns:
      A NumPy array with outliers removed.
  """
  q1 = np.percentile(data, percentile * 100)
  q3 = np.percentile(data, (1 - percentile) * 100)
  iqr = q3 - q1
  lower_bound = q1 - 1.5 * iqr
  upper_bound = q3 + 1.5 * iqr
  return data[(data >= lower_bound) & (data <= upper_bound)]


def replace_outliers_with_neighbor_mean(arr, threshold=8):
    # Calculate the mean and standard deviation of the array
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    
    # Identify outliers
    outliers = np.abs(arr - arr_mean) > threshold * arr_std
    
    # Replace outliers with the mean of their neighboring values
    for i in range(len(arr)):
        if outliers[i]:
            # Find the neighboring values (excluding the outlier itself)
            neighbor_indices = [max(0, i - 1), min(len(arr) - 1, i + 1)]
            neighbor_values = arr[np.logical_and(~outliers, np.isin(range(len(arr)), neighbor_indices))]
            neighbor_mean = np.mean(neighbor_values)
            arr[i] = neighbor_mean
    
    return arr





if __name__ == '__main__':   
    

    # Shapename
    #shape_name = 'Rectangle_2.5_10_5'
    #shape_name = '4x4x8'
    shape_name = 'Cube_5mm'
    
    track_number = 1     # Insert track you want to look at 0-9.
        
    
    column_names = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z']
    track = pd.read_csv('Victor/Project1/' + shape_name + '/Tracks/track' + str(track_number).zfill(2) + '.csv', comment='#', names=column_names)


    # Changing from degrees to radians.
    track[['ori_x', 'ori_y', 'ori_z']] = track[['ori_x', 'ori_y', 'ori_z']].apply(lambda x : np.deg2rad(x), axis=0)
    

    # Plotting position
    ylabels = ['Position x [mm]',
               'Position y [mm]', 
               'Position z [mm]']
    #plot_xyz(track.pos_x, track.pos_y, track.pos_z, ylabels, 'Position of Particle')
    
    # If you would like to filter the signal you could among many things do as below
    # Experiment with the window_length adn polyorder.
    from scipy.signal import savgol_filter
    ori_x_filtered = savgol_filter(track.ori_x, window_length=10, polyorder=1)
    ori_y_filtered = savgol_filter(track.ori_y, window_length=10, polyorder=1)
    ori_z_filtered = savgol_filter(track.ori_z, window_length=10, polyorder=1)
    ylabels = ['Orientation x [rad]',
               'Orientation y [rad]', 
               'Orientation z [rad]']
    #plot_xyz(ori_x_filtered, ori_y_filtered, ori_z_filtered, ylabels, 'Filtered Orientation of Particle')

    # ==================================================================================
    # Now I will show how to interpret this weird rodriguez rotation vector.
    
    # 1.
    # A CAD file consists of a series of triangles. Therefore for a 
    # cuboid each rectangular face consists of 2 triangles and thus in total 12 triangles.
    # Each triangle is defined from 3 points in 3D space. Thus we get 12 triangles each
    # consisting of 3 points with 3 coordinates.
    # For simplicity I have added this as a .npy file which is loaded in the lines below.
    
    shape_file = np.load('Victor/Project1/' + shape_name + '/' + shape_name + '.npy')
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
        rot_matrix = get_rotation_matrix_from_rotvec( np.rad2deg(rodriguez_vector) )
        rotated_vectors = vectors.reshape(-1,3) @ rot_matrix.transpose()
        rotated_vectors = rotated_vectors.reshape(-1,3,3)
        
        # After rotation we move all vectors
        translation_vector = positions[i]
        vectors_rotated_and_translated.append( rotated_vectors + translation_vector )

    
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
    
    print(' ===== Yay you completed this introductory task to moving and rotating shapes. =====')
    
    
    
    # Finding velocities
    # To get velocity from position we need to take the derivate of position. 
    # This can be done numerically using the np.gradient function. 
    # For reference check the documentation https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
    sample_rate = 1.0/100.0     # We are filming at 100 fps so each measurement is space by 1/!00 of a second
    velocities = np.gradient(positions, sample_rate, axis=0)
    velocities = velocities / 1000.0
    plot_xyz(
             velocities[:,0],
             velocities[:,1],
             velocities[:,2],
             ['Velocity x [m/s]',
              'Velocity y [m/s]',
              'Velocity z [m/s]'],
              'Velocity in each direction'
             )

    # Total velocity throug the water

    TotVelocity = np.linalg.norm(velocities, axis = 1)
    plt.figure()
    plt.plot(TotVelocity)

    CleanTotVelocity = replace_outliers_with_neighbor_mean(TotVelocity)
    plt.figure()
    plt.plot(CleanTotVelocity)

    Filt_Tot_Velo = savgol_filter(CleanTotVelocity,window_length=10,polyorder=1)
    plt.figure()
    plt.plot(Filt_Tot_Velo)

    Acceleration = np.diff(Filt_Tot_Velo)
    Filt_Acceleration = savgol_filter(Acceleration,window_length=10,polyorder=1)
    plt.figure()
    plt.plot(Filt_Acceleration)

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
    def relative_rotation_vector(vec1, vec2):
        
        R1 = get_rotation_matrix_from_rotvec(np.rad2deg(vec1))
        R2 = get_rotation_matrix_from_rotvec(np.rad2deg(vec2))
        
        # Compute relative matrix
        R_relative = np.dot(R2, R1.T)
        
        # Convert back to vector
        relative_vec = get_rodriguez_from_rotation_matrix(R_relative)
        
        return relative_vec
    
    relative_to_first = []
    for idx, vec in enumerate(rodriguez_vectors, start = 1):
        relative_to_first.append( relative_rotation_vector(rodriguez_vectors[0], vec) )
    
    relative_to_first = np.array(relative_to_first)
    
    
    plot_xyz(
             relative_to_first[:,0],
             relative_to_first[:,1],
             relative_to_first[:,2],
             ['Orientation x [rad]',
              'Orientation y [rad]',
              'Orientation z [rad]'],
             'Relative to first appearance orientation'     
             )
    
    plt.show()
    
    

    print(stop)
    
    
    
    
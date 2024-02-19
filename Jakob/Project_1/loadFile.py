
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
import scipy

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


if __name__ == '__main__':
    # Loading track as pandas dataframe
    track = pd.read_csv(r'Jakob\Project_1\track00.csv')

    # %%
    # Plotting position
    ylabels = ['Position x [mm]',
               'Position y [mm]', 
               'Position z [mm]']
    plot_xyz(track.pos_x, track.pos_y, track.pos_z, ylabels, 'Position of Particle')

    #integrate the position to get the velocity in the z direction the data is sampled at 100 Hz
    time_z = np.arange(0, len(track)/100, 1/100)
    vel_z = scipy.trapz(track.pos_z/1000,time_z)

    
    # Plotting orientation. The orietation is here defined as the rodriguez rotation vector 
    # needed to be applied to the cube as it appears in the CAD file to end as measured in 
    # frame X.
    ylabels = ['Orientation x [mm]',
               'Orientation y [mm]', 
               'Orientation z [mm]']
    plot_xyz(track.ori_x, track.ori_y, track.ori_z, ylabels, 'Orientation of Particle')
    

    # If you would like to filter the signal you could among many things do as below
    # Experiment with the window_length adn polyorder.
    from scipy.signal import savgol_filter
    ori_x_filtered = savgol_filter(track.ori_x, window_length=10, polyorder=1)
    ori_y_filtered = savgol_filter(track.ori_y, window_length=10, polyorder=1)
    ori_z_filtered = savgol_filter(track.ori_z, window_length=10, polyorder=1)
    
    plot_xyz(ori_x_filtered, ori_y_filtered, ori_z_filtered, ylabels, 'Filtered Orientation of Particle')

    plt.show()
    
    #sys.exit('A plot of the position and orientation should have appeared. Once you have understood the code feel free to remove this line of code at Line 116')
    
    #%% ==================================================================================
    # Now I will show how to interpret this weird rodriguez rotation vector.
    
    # 1. We load the shape and this gives us a "Mesh" object. The important thing for
    # us here is the vectors attribute this is a 12x3x3 array for a cuboid. 
    # A CAD file like these .STL files consists of a series of triangles. Therefore for a 
    # cuboid each rectangular face consists of 2 triangles and thus in total 12 triangles.
    # Each triangle is defined from 3 points in 3D space. Thus we get 12 triangles each
    # consisting of 3 points with 3 coordinates.
    
    if False:   
        # If you have installed the stl package you can set True here to get 
        # access to more information about the shape and some useful methods. It is 
        # however not necesarry.
        from stl import mesh
        shape_file = mesh.Mesh.from_file('./Cube_5mm.STL')
        # Extracting the vectors. MAKE SURE WE CENTER THE SHAPE by subtracting the mean of all points
        vectors = shape_file.vectors - np.mean( shape_file.vectors )
    else:
        shape_file = np.load(r'Jakob\Project_1\cube5mm.npy')
        vectors = shape_file - np.mean(shape_file)
    
    # %%
    # 2. In the track dataframe loaded earlier we have a bunch of positions these are the 
    # coordinates in 3D space of the !!center!! of the particle. To illustrate the path 
    # of our particle falling we can simple take all of the points of our triangles (vectors)
    # and translate them by [pos_x, pos_y, pos_z].
    
    # Extracting the positions
    positions = track[['pos_x', 'pos_y', 'pos_z']].to_numpy()
    
    # Creating a list of vectors for each instance we want to plot
    plot_every_X_instances = 25     # Try modifying this to 1 to get an idea of the sampling frequency.
    vectors_translated = []     
    for i in range(0, len(track), plot_every_X_instances):
        translation_vector = positions[i]
        vectors_translated.append( vectors + translation_vector )
    
    # %%
    # 3. We now have a list of shapes at different points in space. We can then use 
    # matplotlib to plot the position of the shape
    
    # First we create a plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Note we are using projection='3d' since we want a 3D figure
    
    # Now we loop over every shape instance we can find
    for i in range(len(vectors_translated)):
        # Creating a polycollection for this set of triangles
        poly_collection = Poly3DCollection(vectors_translated[i], 
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
    plt.show()
    # Try to open the window and look at the pattern. Notice the secondary motion.
    #sys.exit('You should now have a plot of the falling cube but without rotation. '
    #         'Remove this line of code when you have looked at the plot and want to move on. Line 177')
    # %%
    # 4. In this part I will show how to use the rodriguez vector to the rotate the shape
    # based on the data stored in the columns track.ori_x, track.ori_y, track.ori_z.
    # I have provided a function above that returns the rotation matrix based on a 
    # rodriguez rotation vector. We can extract all the rodriguez vectors for convience
    rodriguez_vectors = track[['ori_x', 'ori_y', 'ori_z']].to_numpy()
    
    # I will start with an example rotating the shape by some amount. Note the "vectors"
    # is centered at (0,0,0). We should rotate the shape before we move it. Rotations occur
    # around the center of the coordinate system used.
    rodriguez_vector_example = rodriguez_vectors[0]
    # To rotate the shape simply apply the rotation matrix to all vectors in vectors.
    # We use reshape to get all the vectors in a 2D matrix (.reshape(-1,3)) and then apply the rotation 
    # matrix to all points. Afterwards we reshape the vector back to (-1,3,3)
    rot_matrix = get_rotation_matrix_from_rotvec( rodriguez_vector_example )
    rotated_vectors = vectors.reshape(-1,3) @ rot_matrix.transpose()
    rotated_vectors = rotated_vectors.reshape(-1,3,3)
    # We can now plot vectors and rotated_vectors next to each other.
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    poly_collection = Poly3DCollection(vectors, 
                                       alpha=1.0, 
                                       facecolors='gray', 
                                       edgecolors='black')
    ax.add_collection(poly_collection)
    ax.set_title('Cube in base orientation')
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_zlim(-7,7)
    ax.set_aspect('equal')
    
    ax = fig.add_subplot(122, projection='3d')
    poly_collection = Poly3DCollection(rotated_vectors, 
                                       alpha=1.0, 
                                       facecolors='gray', 
                                       edgecolors='black')
    ax.add_collection(poly_collection)    
    ax.set_title('Rotated cube')
    ax.set_xlim(-7,7)
    ax.set_ylim(-7,7)
    ax.set_zlim(-7,7)
    ax.set_aspect('equal')
    
    # You should now hopefully see a cube on screen on the left that is not rotated
    # and a cube on the right that is. Yay!!
    #plt.show()
    #sys.exit('You should now have a plot of a cube not rotated and one that is rotated '
    #         'When youre ready to move on delete this line. Line 225')
    
    # %%
    # 5. Now that we know how to rotate and move a cube we can do the exact same loop as 
    # point 2-3. but simply rotate the vectors before moving them. The code could look as 
    # follows
    plot_every_X_instances = 25
    vectors_rotated_and_translated = []     
    for i in range(0, len(track), plot_every_X_instances):
        # First we rotate the shape
        rodriguez_vector = rodriguez_vectors[i]
        # Get rotation matrix and apply to vectors
        rot_matrix = get_rotation_matrix_from_rotvec( rodriguez_vector )
        rotated_vectors = vectors.reshape(-1,3) @ rot_matrix.transpose()
        rotated_vectors = rotated_vectors.reshape(-1,3,3)
        
        # After rotation we move all vectors
        translation_vector = positions[i]
        vectors_rotated_and_translated.append( rotated_vectors + translation_vector )
    
    # First we create a plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Note we are using projection='3d' since we want a 3D figure
    
    # Now we loop over every shape instance we can find
    for i in range(len(vectors_translated)):
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
    
    plt.show()
    print(' ===== Yay you completed this introductory task to moving and rotating shapes. =====')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
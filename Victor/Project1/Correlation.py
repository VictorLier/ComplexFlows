import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

from loadFile_v2 import replace_outliers_with_neighbor_mean

combined_data = []

# Shapename
shape_name = 'Rectangle_2.5_10_5'
#shape_name = '4x4x8'
#shape_name = 'Cube_5mm'

sample_rate = 100
sample_spacing = 1/sample_rate

track_number = 1
data = np.loadtxt('Victor/Project1/' + shape_name + '/Tracks/track0' + str(track_number) + '.csv', skiprows=1, delimiter=",", comments="#")

velocitiy_x = np.gradient(data[:,0],sample_spacing,axis=0)
velocitiy_y = np.gradient(data[:,1],sample_spacing,axis=0)
velocitiy_z = np.gradient(data[:,2],sample_spacing,axis=0)
velocitiy_rotx = np.gradient(data[:,3],sample_spacing,axis=0)
velocitiy_roty = np.gradient(data[:,4],sample_spacing,axis=0)
velocitiy_rotz = np.gradient(data[:,5],sample_spacing,axis=0)

velocitiy_x = replace_outliers_with_neighbor_mean(velocitiy_x, threshold=8)
velocitiy_y = replace_outliers_with_neighbor_mean(velocitiy_y, threshold=8)
velocitiy_z = replace_outliers_with_neighbor_mean(velocitiy_z, threshold=8)
velocitiy_rotx = replace_outliers_with_neighbor_mean(velocitiy_rotx, threshold=8)
velocitiy_roty = replace_outliers_with_neighbor_mean(velocitiy_roty, threshold=8)
velocitiy_rotz = replace_outliers_with_neighbor_mean(velocitiy_rotz, threshold=8)


velocitiy_x  = savgol_filter(velocitiy_x, window_length=10, polyorder=1)
velocitiy_y  = savgol_filter(velocitiy_y, window_length=10, polyorder=1)
velocitiy_z  = savgol_filter(velocitiy_z, window_length=10, polyorder=1)
velocitiy_rotx  = savgol_filter(velocitiy_rotx, window_length=10, polyorder=1)
velocitiy_roty  = savgol_filter(velocitiy_roty, window_length=10, polyorder=1)
velocitiy_rotz  = savgol_filter(velocitiy_rotz, window_length=10, polyorder=1)


Acceleration_x = np.gradient(velocitiy_x,sample_spacing,axis=0)
Acceleration_y = np.gradient(velocitiy_y,sample_spacing,axis=0)
Acceleration_z = np.gradient(velocitiy_z,sample_spacing,axis=0)
Acceleration_rotx = np.gradient(velocitiy_rotx,sample_spacing,axis=0)
Acceleration_roty = np.gradient(velocitiy_roty,sample_spacing,axis=0)
Acceleration_rotz = np.gradient(velocitiy_rotz,sample_spacing,axis=0)

Acceleration_x  = savgol_filter(Acceleration_x, window_length=10, polyorder=1)
Acceleration_y  = savgol_filter(Acceleration_y, window_length=10, polyorder=1)
Acceleration_z  = savgol_filter(Acceleration_z, window_length=10, polyorder=1)
Acceleration_rotx  = savgol_filter(Acceleration_rotx, window_length=10, polyorder=1)
Acceleration_roty  = savgol_filter(Acceleration_roty, window_length=10, polyorder=1)
Acceleration_rotz  = savgol_filter(Acceleration_rotz, window_length=10, polyorder=1)


speed = np.sqrt(velocitiy_x**2 + velocitiy_y**2 + velocitiy_z**2)

#Direction
Vel_direc_x = velocitiy_x/speed
Vel_direc_y = velocitiy_y/speed
Vel_direc_z = velocitiy_z/speed

#Rotation in flow direction coordinates
Orix = data[:,3] - Vel_direc_x
Oriy = data[:,4] - Vel_direc_y
Oriz = data[:,5] - Vel_direc_z

Orix = savgol_filter(Orix, window_length=10, polyorder=1)
Oriy = savgol_filter(Oriy, window_length=10, polyorder=1)
Oriz = savgol_filter(Oriz, window_length=10, polyorder=1)

Orientation = np.stack((Orix,Oriy,Oriz,speed),axis=1)



for track_number in range(9):
    data = np.loadtxt('Victor/Project1/' + shape_name + '/Tracks/track0' + str(track_number) + '.csv', skiprows=1, delimiter=",", comments="#")

    velocitiy_x = np.gradient(data[:,0],sample_spacing,axis=0)
    velocitiy_y = np.gradient(data[:,1],sample_spacing,axis=0)
    velocitiy_z = np.gradient(data[:,2],sample_spacing,axis=0)
    velocitiy_rotx = np.gradient(data[:,3],sample_spacing,axis=0)
    velocitiy_roty = np.gradient(data[:,4],sample_spacing,axis=0)
    velocitiy_rotz = np.gradient(data[:,5],sample_spacing,axis=0)

    velocitiy_x = replace_outliers_with_neighbor_mean(velocitiy_x, threshold=8)
    velocitiy_y = replace_outliers_with_neighbor_mean(velocitiy_y, threshold=8)
    velocitiy_z = replace_outliers_with_neighbor_mean(velocitiy_z, threshold=8)
    velocitiy_rotx = replace_outliers_with_neighbor_mean(velocitiy_rotx, threshold=8)
    velocitiy_roty = replace_outliers_with_neighbor_mean(velocitiy_roty, threshold=8)
    velocitiy_rotz = replace_outliers_with_neighbor_mean(velocitiy_rotz, threshold=8)


    velocitiy_x  = savgol_filter(velocitiy_x, window_length=10, polyorder=1)
    velocitiy_y  = savgol_filter(velocitiy_y, window_length=10, polyorder=1)
    velocitiy_z  = savgol_filter(velocitiy_z, window_length=10, polyorder=1)
    velocitiy_rotx  = savgol_filter(velocitiy_rotx, window_length=10, polyorder=1)
    velocitiy_roty  = savgol_filter(velocitiy_roty, window_length=10, polyorder=1)
    velocitiy_rotz  = savgol_filter(velocitiy_rotz, window_length=10, polyorder=1)


    Acceleration_x = np.gradient(velocitiy_x,sample_spacing,axis=0)
    Acceleration_y = np.gradient(velocitiy_y,sample_spacing,axis=0)
    Acceleration_z = np.gradient(velocitiy_z,sample_spacing,axis=0)
    Acceleration_rotx = np.gradient(velocitiy_rotx,sample_spacing,axis=0)
    Acceleration_roty = np.gradient(velocitiy_roty,sample_spacing,axis=0)
    Acceleration_rotz = np.gradient(velocitiy_rotz,sample_spacing,axis=0)

    Acceleration_x  = savgol_filter(Acceleration_x, window_length=10, polyorder=1)
    Acceleration_y  = savgol_filter(Acceleration_y, window_length=10, polyorder=1)
    Acceleration_z  = savgol_filter(Acceleration_z, window_length=10, polyorder=1)
    Acceleration_rotx  = savgol_filter(Acceleration_rotx, window_length=10, polyorder=1)
    Acceleration_roty  = savgol_filter(Acceleration_roty, window_length=10, polyorder=1)
    Acceleration_rotz  = savgol_filter(Acceleration_rotz, window_length=10, polyorder=1)


    speed = np.sqrt(velocitiy_x**2 + velocitiy_y**2 + velocitiy_z**2)

    #Direction
    Vel_direc_x = velocitiy_x/speed
    Vel_direc_y = velocitiy_y/speed
    Vel_direc_z = velocitiy_z/speed

    #Rotation in flow direction coordinates
    Orix = data[:,3] - Vel_direc_x
    Oriy = data[:,4] - Vel_direc_y
    Oriz = data[:,5] - Vel_direc_z

    Orix = savgol_filter(Orix, window_length=10, polyorder=1)
    Oriy = savgol_filter(Oriy, window_length=10, polyorder=1)
    Oriz = savgol_filter(Oriz, window_length=10, polyorder=1)

    OriWorking = np.stack((Orix,Oriy,Oriz,speed),axis=1)
    Orientation = np.vstack((Orientation, OriWorking))


plt.figure()
plt.plot(Orientation[:,0], label='x')
plt.plot(Orientation[:,1], label='y')
plt.plot(Orientation[:,2], label='z')
plt.plot(Orientation[:,3], label='speed')
plt.legend()


if shape_name == 'Rectangle_2.5_10_5':
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot scatter points with color mapping based on speed
    ax.scatter(Orientation[:,0], Orientation[:,1], Orientation[:,3], c=Orientation[:,3], cmap='viridis', alpha=0.8)

    # Configure plot labels and title
    ax.set_xlabel('Orix')
    ax.set_ylabel('Oriy')
    ax.set_zlabel('Speed')
    ax.set_title('3D Plot with Speed on Z-Axis')


plt.show()


print("Stop")
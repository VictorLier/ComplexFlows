import numpy as np
import matplotlib.pyplot as plt
import math

from loadFile_v2 import replace_outliers_with_neighbor_mean

sample_rate = 100
sample_spacing = 1/sample_rate
# Shapename
shape_name = 'Rectangle_2.5_10_5'
#shape_name = '4x4x8'
#shape_name = 'Cube_5mm'

CdA =[]


#Volume of shape
if shape_name == 'Rectangle_2.5_10_5':
    volume = 2.5*10*5 / 1000000000 #[m^3]
    Area = (2.5*10*2 + 5*10*2 + 5*2.5*2) / 1000000 #[m^2]
    MaxArea = 10*5 / 1000000
    MinimumArea = 2.5*5 / 1000000
elif shape_name == '4x4x8':
    volume = 4*4*8 / 1000000000 #[m^3]
    Area = (8*4*4 + 4*4*2) /  1000000 #[m^2]
    MaxArea = 4*8 / 1000000
    MinimumArea = 4*4 / 1000000  
elif shape_name == 'Cube_5mm':
    volume = 5*5*5 / 1000000000 #[m^3]
    Area = 5*5*6 / 1000000 #[m^2]
    DV = (6 * volume/math.pi)**(1/3) #[m]
    DA = math.sqrt(Area / math.pi)
    MaxArea = 5*5 / 1000000 #[m]
    MinimumArea = 5*5 / 1000000 #[m]

DV = (6 * volume/math.pi)**(1/3) #[m]
DA = math.sqrt(Area / math.pi)

Cross_Area_Volume = (DV/2)**2 * math.pi
Cross_Area_Area = (DA/2)**2 * math.pi

#mass of shape
rho_shape = 1030 #[kg/m^3]
mass_shape = rho_shape * volume #[kg]
#Water density
rho_water = 997.877 #[kg/m^3]
#Change in potential energy
g = 9.82




for track_number in range(10):
    #track_number = 0

    #column_names = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z']
    data = np.loadtxt('Victor/Project1/' + shape_name + '/Tracks/track0' + str(track_number) + '.csv', skiprows=1, delimiter=",", comments="#")

    #total z change
    tot_z = data[0,2] - data[-1,2] #[mm]
    tot_z = tot_z / 1000 #[m]

    #Total time from 0 - total z
    tot_time = len(data[:,0])/sample_rate #[s]

    #Total distance through water
    distances = np.sqrt(np.sum(np.diff(data[:,:3], axis=0)**2, axis=1))
    tot_distance = np.sum(distances) / 1000 #[m]

    #Change in potential energy
    dPE = (mass_shape - volume*rho_water) * g * tot_z #[J]

    #speed
    speed = tot_distance/tot_time #[m/s]

    #Force required
    Force = dPE/tot_distance

    #Drag force
    #F = 1/2 * rho * A * cd * v^2
    CdA.append((2 * Force) / (rho_water * speed**2))

plt.figure()
plt.plot(CdA)
plt.show()

CdA_mean = np.mean(CdA)
CD_volume = CdA_mean/Cross_Area_Volume
CD_Area = CdA_mean/Cross_Area_Area
CD_MaxArea = CdA_mean/MaxArea
CD_MinArea = CdA_mean/MinimumArea

CdA_std = np.std(CdA)
CD_volume_error = CdA_std/Cross_Area_Volume
CD_Area_error = CdA_std/Cross_Area_Area
CD_MaxArea_error = CdA_std/MaxArea
CD_MinArea_error = CdA_std/MinimumArea



print(CD_volume, CD_Area, CD_MaxArea, CD_MinArea)
print(CD_volume_error, CD_Area_error, CD_MaxArea_error, CD_MinArea_error)



print("stop")
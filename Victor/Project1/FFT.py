import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import math

from loadFile_v2 import replace_outliers_with_neighbor_mean


sample_rate = 100
sample_spacing = 1/sample_rate
# Shapename
#shape_name = 'Rectangle_2.5_10_5'
#shape_name = '4x4x8'
shape_name = 'Cube_5mm'

fft_plot_x = []
fft_plot_y = []
fft_plot_z = []
fft_plot_rotx = []
fft_plot_roty = []
fft_plot_rotz = []


for track_number in range(10):
    #track_number = 3
    #column_names = ['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z']
    data = np.loadtxt('Victor/Project1/' + shape_name + '/Tracks/track0' + str(track_number) + '.csv', skiprows=1, delimiter=",", comments="#")
    '''
    for i in range(6):
        velocity = np.gradient(data[:,i],sample_spacing,axis=0)

        velocity = replace_outliers_with_neighbor_mean(velocity)    

        #Padding
        padded_length = 2 ** int(math.ceil(math.log2(len(velocitiy))))
        velocitiy = np.pad(velocitiy, (0, padded_length - len(velocitiy)), mode='constant')

        fft_output = np.fft.rfft(velocitiy)
        abs_fft = abs(fft_output)
        power_spectrum = abs_fft**2
    '''

        

    velocitiy_x = np.gradient(data[:,0],sample_spacing,axis=0)
    velocitiy_y = np.gradient(data[:,1],sample_spacing,axis=0)
    velocitiy_z = np.gradient(data[:,2],sample_spacing,axis=0)
    velocitiy_rotx = np.gradient(data[:,3],sample_spacing,axis=0)
    velocitiy_roty = np.gradient(data[:,4],sample_spacing,axis=0)
    velocitiy_rotz = np.gradient(data[:,5],sample_spacing,axis=0)

    #velocitiy_rotx = data[:,3]
    #velocitiy_roty = data[:,4]
    #velocitiy_rotz = data[:,5]


    #plt.figure()
    #plt.plot(velocitiy_rotz)
    #plt.show()


    velocitiy_x = replace_outliers_with_neighbor_mean(velocitiy_x)
    velocitiy_y = replace_outliers_with_neighbor_mean(velocitiy_y)
    velocitiy_z = replace_outliers_with_neighbor_mean(velocitiy_z)
    velocitiy_rotx = replace_outliers_with_neighbor_mean(velocitiy_rotx)
    velocitiy_roty = replace_outliers_with_neighbor_mean(velocitiy_roty)
    velocitiy_rotz = replace_outliers_with_neighbor_mean(velocitiy_rotz)

    #plt.figure()
    #plt.plot(velocitiy_rotz)
    #plt.show()


    velocitiy_x = savgol_filter(velocitiy_x, window_length=10, polyorder=1)
    velocitiy_y = savgol_filter(velocitiy_y, window_length=10, polyorder=1)
    velocitiy_z = savgol_filter(velocitiy_z, window_length=10, polyorder=1)
    velocitiy_rotx = savgol_filter(velocitiy_rotx, window_length=10, polyorder=1)
    velocitiy_roty = savgol_filter(velocitiy_roty, window_length=10, polyorder=1)
    velocitiy_rotz = savgol_filter(velocitiy_rotz, window_length=10, polyorder=1)

    #Padding
    padded_length_x = 2 ** int(math.ceil(math.log2(len(velocitiy_x))))
    padded_length_y = 2 ** int(math.ceil(math.log2(len(velocitiy_y))))
    padded_length_z = 2 ** int(math.ceil(math.log2(len(velocitiy_z))))
    padded_length_rotx = 2 ** int(math.ceil(math.log2(len(velocitiy_rotx))))
    padded_length_roty = 2 ** int(math.ceil(math.log2(len(velocitiy_roty))))
    padded_length_rotz = 2 ** int(math.ceil(math.log2(len(velocitiy_rotz))))

    velocitiy_x = np.pad(velocitiy_x, (0, padded_length_x - len(velocitiy_x)), mode='constant')
    velocitiy_y = np.pad(velocitiy_y, (0, padded_length_y - len(velocitiy_y)), mode='constant')
    velocitiy_z = np.pad(velocitiy_z, (0, padded_length_z - len(velocitiy_z)), mode='constant')
    velocitiy_rotx = np.pad(velocitiy_rotx, (0, padded_length_rotx - len(velocitiy_rotx)), mode='constant')
    velocitiy_roty = np.pad(velocitiy_roty, (0, padded_length_roty - len(velocitiy_roty)), mode='constant')
    velocitiy_rotz = np.pad(velocitiy_rotz, (0, padded_length_rotz - len(velocitiy_rotz)), mode='constant')

    fft_output_x = np.fft.fft(velocitiy_x)
    fft_output_y = np.fft.fft(velocitiy_y)
    fft_output_z = np.fft.fft(velocitiy_z)
    fft_output_rotx = np.fft.fft(velocitiy_rotx)
    fft_output_roty = np.fft.fft(velocitiy_roty)
    fft_output_rotz = np.fft.fft(velocitiy_rotz)

    abs_fft_x = abs(fft_output_x)
    abs_fft_y = abs(fft_output_y)
    abs_fft_z = abs(fft_output_z)
    abs_fft_rotx = abs(fft_output_rotx)
    abs_fft_roty = abs(fft_output_roty)
    abs_fft_rotz = abs(fft_output_rotz)

    power_spectrum_x = abs_fft_x**2
    power_spectrum_y = abs_fft_y**2
    power_spectrum_z = abs_fft_z**2
    power_spectrum_rotx = abs_fft_rotx**2
    power_spectrum_roty = abs_fft_roty**2
    power_spectrum_rotz = abs_fft_rotz**2

    fft_plot_x.append(power_spectrum_x)
    fft_plot_y.append(power_spectrum_y)
    fft_plot_z.append(power_spectrum_z)
    fft_plot_rotx.append(power_spectrum_rotx)
    fft_plot_roty.append(power_spectrum_roty)
    fft_plot_rotz.append(power_spectrum_rotz)

#data = savgol_filter(data, window_length=10, polyorder=1)
        
#Windowing
#window = np.hanning(len(data))
#data = data * window
        
#Padding
#padded_length = 2 ** int(math.ceil(math.log2(len(data))))
        
#data = np.pad(data, (0, padded_length - len(data)), mode='constant')
        
#fft_output = np.fft.rfft(data)
#abs_fft = abs(fft_output)
#power_spectrum = abs_fft**2


#mean_fft = np.mean(fft_final,axis=1)

fft_plot_x = np.mean(fft_plot_x,axis=0)
fft_plot_y = np.mean(fft_plot_y,axis=0)
fft_plot_z = np.mean(fft_plot_z,axis=0)
fft_plot_rotx = np.mean(fft_plot_rotx,axis=0)
fft_plot_roty = np.mean(fft_plot_roty,axis=0)
fft_plot_rotz = np.mean(fft_plot_rotz,axis=0)

freqs = np.fft.fftfreq(len(velocitiy_x),sample_spacing)

    

# Plot the spectrum
#plt.figure()
#plt.loglog(freqs, fft_plot_x, label='x')
#plt.loglog(freqs, fft_plot_y, label='y')
#plt.loglog(freqs, fft_plot_z, label='z')
#plt.loglog(freqs, fft_plot_rotx, label='rotx')
#plt.loglog(freqs, fft_plot_roty, label='roty')
#plt.loglog(freqs, fft_plot_rotz, label='rotz')
#plt.legend()
#plt.xlabel("time (s)")
#plt.ylabel("Voltage (mV)")
#plt.title(shape_name)

plt.figure()
plt.loglog(fft_plot_x, label='x')
plt.loglog(fft_plot_y, label='y')
plt.loglog(fft_plot_z, label='z')
plt.loglog(fft_plot_rotx, label='rotx')
plt.loglog(fft_plot_roty, label='roty')
plt.loglog(fft_plot_rotz, label='rotz')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("Voltage (mV)")
plt.title(shape_name)




plt.show()

#import tikzplotlib

#tikzplotlib.save("Victor/Project1/test.tex")

print("Stop")


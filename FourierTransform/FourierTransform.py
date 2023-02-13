import numpy as np

import matplotlib.pyplot as plotter

# How many time points are needed

samplingFrequency = 1000

# At what intervals time points are sampled

samplingInterval = 1/samplingFrequency

# Start time of the signals

startTime = -50

# Begin time period of the signals

beginTime = 0

# Period time of the signals

periodTime = 1

# End time of the signals

endTime = 50

# Time points

nonTimepointsStart = np.arange(startTime, beginTime, samplingInterval)

periodTimepoints = np.arange(beginTime, periodTime, samplingInterval)

nonTimepointsEnd = np.arange(periodTime, endTime, samplingInterval)

Timepoints = np.arange(startTime, endTime, samplingInterval)

# Define the constant function

amplitude =[0 for i in nonTimepointsStart] +  [1 for i in periodTimepoints] + [0 for i in nonTimepointsEnd] 

# Create a subplot

figure, axis = plotter.subplots(3,1)

plotter.subplots_adjust(hspace=1)

figure1 = plotter.figure()

# Time domain representation

axis[0].set_title('Function time domain ')

axis[0].plot(Timepoints, amplitude)

axis[0].set_xlabel('Time')

axis[0].set_ylabel('Amplitude')

# axis[0].set_xlim((-5,5))

# Frequency domain representation

tpCount     = len(amplitude)       #N

''' 

0,1,2,3....N-1/2 (k)

f = (k/N)*fs Convert from discrete to continous, discrete considers (k/N) as f.

'''

frequencies = np.fft.fftfreq(int(tpCount),1/samplingFrequency)

frequencies = np.fft.fftshift(frequencies)

# Energy Spectrum Frequency domain representation

fourierTransform = (np.fft.fft(amplitude)/samplingFrequency)

fourierTransform = np.fft.fftshift(fourierTransform)

axis1 = figure1.add_subplot(1,1,1,projection='3d')

axis1.set_title('Fourier Transform')

axis1.plot(frequencies, np.real(fourierTransform),np.imag(fourierTransform))

print("fourier transform:",fourierTransform[int(len(fourierTransform)/2)+1000])

axis1.set_xlabel('Frequency')

axis1.set_ylabel('Real part')

axis1.set_zlabel('Imag part')

axis1.set_xlim([-50, 50])

# Energy spectrum dB representation

axis[1].set_title('Energy spectrum 3dB bandwidth and 40dB bandwidth')

energySpectrum = abs(fourierTransform)**2

maxEnergySpectrum = max(abs(fourierTransform)**2)

dBenergySpectrum = 10*np.log10(abs(fourierTransform)**2)

axis[1].plot(frequencies, dBenergySpectrum)

axis[1].set_xlabel('Frequency')

axis[1].set_ylabel('Amplitude')

axis[1].set_ylim((-60,None))

axis[1].set_xlim([-50, 50])

# Calculate the cut off 40dB frequency

for i in range(1,int(tpCount/2)):

    if energySpectrum[-i]/maxEnergySpectrum > 1/10000:

        id40 = int(tpCount)-i

        print("frequencies[id40]: ",id40)

        axis[1].plot(frequencies[id40], dBenergySpectrum[id40],'ro')

        axis[1].text(frequencies[id40], dBenergySpectrum[id40]+0.001, '%.2f' % frequencies[id40], ha='center', va= 'bottom',fontsize=9)

        break
    
# Calculate the cut off 3dB frequency

for i in range(1,int(tpCount/2)):

    if energySpectrum[-i]/maxEnergySpectrum > 0.501:

        id3 = int(tpCount)-i
        
        axis[1].plot(frequencies[id3], dBenergySpectrum[id3],'ro')
    
        axis[1].text(frequencies[id3], dBenergySpectrum[id3]+0.001, '%.2f' % frequencies[id3], ha='center', va= 'bottom',fontsize=9)

        break

# Calculate Percentage of Total Energy Integral Bandwidth

frequencyResolution = frequencies[1] - frequencies[0]

frequencies = frequencies[int(tpCount/2):]

energySpectrum = energySpectrum[int(tpCount/2):]

outputEnergy = [0]*int(tpCount/2)

outputEnergy[0] = 0

for i in range(1,int(tpCount/2)):

    outputEnergy[i] = (energySpectrum[i] + energySpectrum[i-1])*frequencyResolution/2 + outputEnergy[i-1]

outputEnergyPer = outputEnergy/outputEnergy[-1]

print("outputEnergyPer: ",frequencies)

for i in range(0,int(tpCount/2)):

    if outputEnergyPer[i] >= 0.98:

        p98 = i
        
        axis[2].plot(frequencies[p98], outputEnergyPer[p98],'ro')
    
        axis[2].text(frequencies[p98], outputEnergyPer[p98]+0.001, '%.2f' % frequencies[p98], ha='center', va= 'bottom',fontsize=9)

        break

# Has some error because of the trapezoidal rule

axis[2].plot(frequencies, outputEnergyPer)

#axis[2].set_xlim([0, 2])

axis[2].set_title('Output Energy ')

axis[2].set_xlabel('Integral Bandwidth Frequency ')

axis[2].set_ylabel('Percentage ')

plotter.show()
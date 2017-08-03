# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 08:47:47 2016
@name: test_2.py
@description:
    - Acquiring and extracting signals to test datasets.py module
    - Training use ANN and estimate accuracy of system
@author: VPi
"""

import time
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

import csv
import ANN
import initialization as init

'''
Function: hanning_window(hanning_window)
Description: apply hanning window for signal
'''
def hanning_window(hanning_window):
    # Apply hanning window for signal
    return np.multiply(np.hanning(256), hanning_window)

'''
Function: IIR_filter(signal_filter)
Description: Filter signal to get frequency spectrum 8 - 3Hz
'''
def IIR_filter(signal_filter):
    after_filter = []
    # Create IIR Butterworth filter
    # bandpass from 8Hz to 30Hz
    b, a = signal.butter(4, [0.125, 0.46875], btype = 'bandpass')
    
    # Apply IIR filter to signal
    after_filter = signal.filtfilt(b, a, signal_filter)
    
    return after_filter

'''
Function: energy_calculation(rhythm, N, WINDOW_SIGNAL)
Description: Calculate energy of signal
Variables:
    rhythm = 1: Calculate energy of alpha
    rhythm = 2: Calculate energy of beta
    rhythm = 3: Calculate energy of whole of signal
'''
def energy_calculation(rhythm, N, WINDOW_SIGNAL, freq):
    E = 0
    
    # Energy of alpha rhythm
    if rhythm==1:
        for m in range(0,int(N/2)):
            if freq[m]>=8 and freq[m]<=14:
                E = E + WINDOW_SIGNAL[m]**2
        E = E/N
    # Energy of beta rhythm
    elif rhythm==2:
        for m in range(0,int(N/2)):
            if freq[m]>=14 and freq[m]<=30:
                E = E + WINDOW_SIGNAL[m]**2
        E = E/N   
    # Energy of whole of signal
    elif rhythm==3:
        for m in range(0,int(N/2)):
            E = E + WINDOW_SIGNAL[m]**2
        E = E/N
    
    return E

'''
Function: feature_extraction(ACQ_SIGNAL, WINDOW_SIGNAL, temp_features, BUFFER_FEATURES, INPUT_DATASET, OUTPUT_DATASET)
Description: extract features from acquired signal and store features into INPUT_DATASET
             and create OUTPUT_DATASET at the same time
Variables: global or reference variables
    ACQ_SIGNAL: store signal from 14 electrodes during 10s
    WINDOW_SIGNAL: buffer to store signal with 256 (or 128) data to process (filter, fft, calculate energy)
    temp_features: store temprotary features
    BUFFER_FEATURES: store 42 features from 14 electrodes
    INPUT_DATASET: store each 1025 (or 1152) BUFFER_FEATURES vector
    OUTPUT_DATASET: store 1025 (or 1152) state corresponding to state
    state: support OUTPUT_DATASET
'''
def feature_extraction(state, acc):
    # reference parameter
    ACQ_SIGNAL = []
    WINDOW_SIGNAL = []
    temp_features = 0
    BUFFER_FEATURES = []
    INPUT_DATASET = []
    OUTPUT_DATASET = []
    # Initialize sampling rate and sampling frequency
    f = 128.0 # Sampling rate
    T = 1.0/128.0 # Sampling time
    t = np.linspace(1.0, 256.0, 256) # Sampling time
    t = np.divide(t, f)
    # Number of datas
    N = 256.0
    # Loop variables    
    i = 0
    j = 0
    # Acquire signals by calling acquire_data()   
    if state==1:
        print 'START COLLECTING UP SIGNALS'
        with open('data_test/up-1-27.10.16.17.11.30.csv') as csvfile1:
            readCSV1 = csv.reader(csvfile1, delimiter = ',')
        
            for row in readCSV1:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
    elif state==2:
        print 'START COLLECTING RIGHT SIGNALS'
        with open('data_test/right-1-27.10.16.17.10.42.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
    elif state==3:
        print 'START COLLECTING DOWN SIGNALS'
        with open('data_test/down-2-27.10.16.16.59.27.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
    else:
        print 'START COLLECTING LEFT SIGNALS'
        with open('data_test/left-1-27.10.16.17.07.04.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
    
    ACQ_SIGNAL = np.array(ACQ_SIGNAL, dtype = float)
    
    print 'START PROCESSING'
    while i<1025:
        BUFFER_FEATURES = []
        
        for j in range (0, 14):
            WINDOW_SIGNAL = []
            # Get rectangle window from ACQ_SIGNAL for each electrode                        
            WINDOW_SIGNAL = ACQ_SIGNAL.T[j][i:i+256]
            
            # Apply IIR filter to get frequency from 8Hz to 30Hz
            WINDOW_SIGNAL = IIR_filter(WINDOW_SIGNAL)
            
            # Get hanning window
            WINDOW_SIGNAL = hanning_window(WINDOW_SIGNAL)
            
            # Apply FFT to get spectrum
            WINDOW_SIGNAL = fft(WINDOW_SIGNAL)
            WINDOW_SIGNAL = 2.0/N*np.abs(WINDOW_SIGNAL[0:int(N/2)])
            freq = np.linspace(0.0, 1.0/(2*T), int(N/2))
            
            # Calculate energy alpha and store into temp_feature before storing into BUFFER_FEATURES
            temp_features = energy_calculation(1, N, WINDOW_SIGNAL, freq)
            BUFFER_FEATURES.append(temp_features)
            
            # Calculate energy beta and store into temp_feature before storing into BUFFER_FEATURES
            temp_features = energy_calculation(2, N, WINDOW_SIGNAL, freq)
            BUFFER_FEATURES.append(temp_features)
            
            # Calculate energy whole of signal and store into temp_feature before storing into BUFFER_FEATURES
            temp_features = energy_calculation(3, N, WINDOW_SIGNAL, freq)
            BUFFER_FEATURES.append(temp_features)
            
        # Update features for INPUT_DATASET by adding BUFFER_FEATURE to it
        INPUT_DATASET.append(BUFFER_FEATURES)
        
        # Update action for OUTPUT_DATASET by adding act vector to it
        # Up state
        if state==1:
            OUTPUT_DATASET.append([1,0,0,0])
        # Right state
        elif state==2:
            OUTPUT_DATASET.append([0,1,0,0])
        # Down state
        elif state==3:
            OUTPUT_DATASET.append([0,0,1,0])
        # Left state
        elif state==4:
            OUTPUT_DATASET.append([0,0,0,1])
    
        # Update i
        i = i + acc    
    
    print 'END PROCESSING'
    print
    return INPUT_DATASET, OUTPUT_DATASET

startTime = time.clock()    # Get start time
# Add UP state to datasets
init.input_temp, init.output_temp = feature_extraction(1, 1)

init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
if init.INPUT_DATASETs.shape[0] is not 0:
    init.INPUT_DATASETs = np.concatenate((init.INPUT_DATASETs, init.input_temp), axis = 0)
    init.OUTPUT_DATASETs = np.concatenate((init.OUTPUT_DATASETs, init.output_temp), axis = 0)
else:
    init.INPUT_DATASETs = init.input_temp
    init.OUTPUT_DATASETs = init.output_temp

# Add RIGHT state to datasets
init.input_temp, init.output_temp = feature_extraction(2, 1)

init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
if init.INPUT_DATASETs.shape[0] is not 0:
    init.INPUT_DATASETs = np.concatenate((init.INPUT_DATASETs, init.input_temp), axis = 0)
    init.OUTPUT_DATASETs = np.concatenate((init.OUTPUT_DATASETs, init.output_temp), axis = 0)
else:
    init.INPUT_DATASETs = init.input_temp
    init.OUTPUT_DATASETs = init.output_temp

# Add DOWN state to datasets
init.input_temp, init.output_temp = feature_extraction(3, 1)

init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
if init.INPUT_DATASETs.shape[0] is not 0:
    init.INPUT_DATASETs = np.concatenate((init.INPUT_DATASETs, init.input_temp), axis = 0)
    init.OUTPUT_DATASETs = np.concatenate((init.OUTPUT_DATASETs, init.output_temp), axis = 0)
else:
    init.INPUT_DATASETs = init.input_temp
    init.OUTPUT_DATASETs = init.output_temp

# Add LEFT state to datasets
init.input_temp, init.output_temp = feature_extraction(4, 1)

init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
if init.INPUT_DATASETs.shape[0] is not 0:
    init.INPUT_DATASETs = np.concatenate((init.INPUT_DATASETs, init.input_temp), axis = 0)
    init.OUTPUT_DATASETs = np.concatenate((init.OUTPUT_DATASETs, init.output_temp), axis = 0)
else:
    init.INPUT_DATASETs = init.input_temp
    init.OUTPUT_DATASETs = init.output_temp

print
print init.INPUT_DATASETs.shape
print init.OUTPUT_DATASETs.shape
print

count = 0
for row in init.INPUT_DATASETs:
    for t in range(0, 42):
        if row[t] > 1.0:
            print row[t]
            print 'bullshit'
            count += 1
            
if count ==0:
    print 'May be it good'

''' Training ANN '''
NN = ANN.Neural_Network(Lambda = 0.0001)
T = ANN.trainer(NN)
T.train(init.INPUT_DATASETs, init.OUTPUT_DATASETs)

endTime = time.clock()  # Get end time
# Calculate processing time
processing_time = startTime - endTime

''' Calculate accuracy '''
count1 = 0
count2 = 0
count3 = 0
count4 = 0
count = 0
test = NN.foward(init.INPUT_DATASETs)
# Accuracy of UP state
for t in range(0, test.shape[0]/4):
    if test[t][0] == np.max(test[t]):
        count1 += 1

# Accuracy of RIGHT state
for t in range(test.shape[0]/4, test.shape[0]/2):
    if test[t][1] == np.max(test[t]):
        count2 += 1
        
# Accuracy of DOWN state
for t in range(test.shape[0]/2, test.shape[0]*3/4):
    if test[t][2] == np.max(test[t]):
        count3 += 1

# Accuracy of LEFT state
for t in range(test.shape[0]*3/4, test.shape[0]):
    if test[t][3] == np.max(test[t]):
        count4 += 1

count = count1 + count2 + count3 + count4

print
print 'Accuracy of UP state: ' + str(count1*100.0*4.0/test.shape[0])
print 'Accuracy of RIGHT state: ' + str(count2*100.0*4.0/test.shape[0])
print 'Accuracy of DOWN state: ' + str(count3*100.0*4.0/test.shape[0])
print 'Accuracy of LEFT state: ' + str(count4*100.0*4.0/test.shape[0])
print
print 'Accuracy of whole state: ' + str(count*100.0/test.shape[0])
print 
print 'Processing time is: ' + str(abs(processing_time)) + 's'
print

''' Draw training data, relation between T and error E '''
plt.figure(1)
plt.plot(T.E, label = 'Train line', linewidth = 2.0)
plt.legend()

plt.grid(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

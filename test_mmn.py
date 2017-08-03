# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:23:47 2016
@name: test_mmn.py
@description: Test accuracy of Modular Multi - layer Nets
@author: VPi
"""

import time
import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

import csv
import ANN_MMN as ANN
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
        #E = E/N
    # Energy of beta rhythm
    elif rhythm==2:
        for m in range(0,int(N/2)):
            if freq[m]>=14 and freq[m]<=30:
                E = E + WINDOW_SIGNAL[m]**2
        #E = E/N   
    # Energy of whole of signal
    elif rhythm==3:
        for m in range(0,int(N/2)):
            E = E + WINDOW_SIGNAL[m]**2
        #E = E/N
    
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
def feature_extraction(state, acc, mode):
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
    if state==5:
        print 'START COLLECTING NEUTRAL SIGNALS'
        with open('data_test/neural-1-16.09.16.10.39.50.csv') as csvfile1:
            readCSV1 = csv.reader(csvfile1, delimiter = ',')
        
            for row in readCSV1:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        if mode==1:
            ACQ_SIGNAL = ACQ_SIGNAL[0:3000]
        else:
            ACQ_SIGNAL = ACQ_SIGNAL[1280:2560]
    if state==1:
        print 'START COLLECTING UP SIGNALS'
        with open('data_test/up-1-27.10.16.17.11.30.csv') as csvfile1:
            readCSV1 = csv.reader(csvfile1, delimiter = ',')
        
            for row in readCSV1:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        if mode==1:
            ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
        else:
            ACQ_SIGNAL = ACQ_SIGNAL[1280:2560]
    elif state==2:
        print 'START COLLECTING RIGHT SIGNALS'
        with open('data_test/right-1-27.10.16.17.10.42.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        if mode==1:
            ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
        else:
            ACQ_SIGNAL = ACQ_SIGNAL[1280:2560]
    elif state==3:
        print 'START COLLECTING DOWN SIGNALS'
        with open('data_test/down-2-27.10.16.16.59.27.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        if mode==1:
            ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
        else:
            ACQ_SIGNAL = ACQ_SIGNAL[1280:2560]
    elif state==4:
        print 'START COLLECTING LEFT SIGNALS'
        with open('data_test/left-1-27.10.16.17.07.04.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter = ',')
        
            for row in readCSV:
                ACQ_SIGNAL.append(row[2: 16])
        
        del ACQ_SIGNAL[0]
        if mode==1:
            ACQ_SIGNAL = ACQ_SIGNAL[0:1280]
        else:
            ACQ_SIGNAL = ACQ_SIGNAL[1280:2560]
    
    ACQ_SIGNAL = np.array(ACQ_SIGNAL, dtype = float)
    
    print 'START PROCESSING'
    while (i+255)<1280:
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
            OUTPUT_DATASET.append([0,1])
        # Right state
        elif state==2:
            OUTPUT_DATASET.append([0,1])
        # Down state
        elif state==3:
            OUTPUT_DATASET.append([0,1])
        # Left state
        elif state==4:
            OUTPUT_DATASET.append([0,1])
        elif state==5:
            OUTPUT_DATASET.append([1,0])
    
        # Update i
        i = i + acc    
    
    print 'END PROCESSING'
    print
    return INPUT_DATASET, OUTPUT_DATASET

'''

'''
test_up = np.array([], dtype = float)
test_right = np.array([], dtype = float)
test_down = np.array([], dtype = float)
test_left = np.array([], dtype = float)

# Add NEUTRAL state to datasets
init.input_temp, init.output_temp = feature_extraction(5, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
    
init.UP_INPUT = init.input_temp
init.RIGHT_INPUT = init.input_temp
init.DOWN_INPUT = init.input_temp
init.LEFT_INPUT = init.input_temp

init.UP_OUTPUT = init.output_temp
init.RIGHT_OUTPUT =init.output_temp
init.DOWN_OUTPUT = init.output_temp
init.LEFT_OUTPUT = init.output_temp

# Add UP state to datasets
init.input_temp, init.output_temp = feature_extraction(1, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
    
init.UP_INPUT = np.concatenate((init.UP_INPUT, init.input_temp), axis = 0)
#init.UP_INPUT = init.UP_INPUT/10.0
init.UP_INPUT = np.divide(init.UP_INPUT, 500.0)
init.UP_OUTPUT = np.concatenate((init.UP_OUTPUT, init.output_temp), axis = 0)
    
# Add RIGHT state to datasets
init.input_temp, init.output_temp = feature_extraction(2, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
    
init.RIGHT_INPUT = np.concatenate((init.RIGHT_INPUT, init.input_temp), axis = 0)
#init.RIGHT_INPUT = init.RIGHT_INPUT/10.0
init.RIGHT_INPUT = np.divide(init.RIGHT_INPUT, 500.0)
init.RIGHT_OUTPUT = np.concatenate((init.RIGHT_OUTPUT, init.output_temp), axis = 0)

#test_right = init.input_temp
    
# Add DOWN state to datasets
init.input_temp, init.output_temp = feature_extraction(3, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
    
init.DOWN_INPUT = np.concatenate((init.DOWN_INPUT, init.input_temp), axis = 0)
#init.DOWN_INPUT = init.DOWN_INPUT/10.0
init.DOWN_INPUT = np.divide(init.DOWN_INPUT, 500.0)
init.DOWN_OUTPUT = np.concatenate((init.DOWN_OUTPUT, init.output_temp), axis = 0)

#test_down= init.input_temp
    
# Add LEFT state to datasets
init.input_temp, init.output_temp = feature_extraction(4, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)
    
init.LEFT_INPUT = np.concatenate((init.LEFT_INPUT, init.input_temp), axis = 0)
init.LEFT_INPUT = np.divide(init.LEFT_INPUT, 500.0)
init.LEFT_OUTPUT = np.concatenate((init.LEFT_OUTPUT, init.output_temp), axis = 0)

#test_left = init.input_temp



''' Training ANN '''
NN_UP = ANN.Neural_Network(Lambda = 0.0001)
T_UP = ANN.trainer(NN_UP)
T_UP.train(init.UP_INPUT, init.UP_OUTPUT)
            
# RIGHT Neural Nets
NN_RIGHT = ANN.Neural_Network(Lambda = 0.0001)
T_RIGHT = ANN.trainer(NN_RIGHT)
T_RIGHT.train(init.RIGHT_INPUT, init.RIGHT_OUTPUT)
            
# DOWN Neural Nets
NN_DOWN = ANN.Neural_Network(Lambda = 0.0001)
T_DOWN = ANN.trainer(NN_DOWN)
T_DOWN.train(init.DOWN_INPUT, init.DOWN_OUTPUT)
            
# LEFT Neural Nets
NN_LEFT = ANN.Neural_Network(Lambda = 0.0001)
T_LEFT = ANN.trainer(NN_LEFT)
T_LEFT.train(init.LEFT_INPUT, init.LEFT_OUTPUT)

# Up training line
plt.subplot(221)
plt.plot(T_UP.E)
plt.title('Training line of UP state')
plt.grid(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
        
# Right training line
plt.subplot(223)
plt.plot(T_RIGHT.E)
#plt.title('Training line of RIGHT state')
plt.grid(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
        
# Down training line
plt.subplot(222)
plt.plot(T_DOWN.E)
plt.title('Training line of DOWN state')
plt.grid(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
        
# Left training line
plt.subplot(224)
plt.plot(T_LEFT.E)
plt.title('Training line of LEFT state')
plt.grid(1)
plt.xlabel('Epochs')
plt.ylabel('Cost')
                
plt.show()

''' CHECKING VARIABLES '''
''' CHECKING UP '''
up_state = []
# Test the first max layer
up_flag = []
right_flag = []
down_flag = []
left_flag = []

acc1 = 0
acc2 = 0
acc3 = 0
acc4 = 0

'''
######################################################
#                   TEST UP STATE                    #
######################################################
'''

''' Test '''
init.input_temp, init.output_temp = feature_extraction(1, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
test_up = init.input_temp
test_up = np.divide(test_up, 500.0)

ff_up = NN_UP.foward(test_up)
ff_right = NN_RIGHT.foward(test_up)
ff_down = NN_DOWN.foward(test_up)
ff_left = NN_LEFT.foward(test_up)

# The first max layer
for ff in ff_up:
    if ff[0] <= ff[1]:
        up_flag.append(ff[1])
    else:
        up_flag.append(0)

for ff in ff_right:
    if ff[0] <= ff[1]:
        right_flag.append(ff[1])
    else:
        right_flag.append(0)

for ff in ff_down:
    if ff[0] <= ff[1]:
        down_flag.append(ff[1])
    else:
        down_flag.append(0)

for ff in ff_left:
    if ff[0] <= ff[1]:
        left_flag.append(ff[1])
    else:
        left_flag.append(0)

# The second max layer
up_state.append(up_flag)
up_state.append(right_flag)
up_state.append(down_flag)
up_state.append(left_flag)
up_state = np.array(up_state, dtype = float)
up_state = up_state.T

up_count = 0
for st in up_state:
    if st[0] == np.max(st):
        up_count += 1
    else:
        pass
acc1 = up_count

'''
######################################################
#                   TEST RIGHT STATE                 #
######################################################
'''
up_state = []
# Test the first max layer
up_flag = []
right_flag = []
down_flag = []
left_flag = []

''' Test '''
init.input_temp, init.output_temp = feature_extraction(2, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
test_up = init.input_temp
test_up = np.divide(test_up, 500.0)

ff_up = NN_UP.foward(test_up)
ff_right = NN_RIGHT.foward(test_up)
ff_down = NN_DOWN.foward(test_up)
ff_left = NN_LEFT.foward(test_up)

# The first max layer
for ff in ff_up:
    if ff[0] <= ff[1]:
        up_flag.append(ff[1])
    else:
        up_flag.append(0)

for ff in ff_right:
    if ff[0] <= ff[1]:
        right_flag.append(ff[1])
    else:
        right_flag.append(0)

for ff in ff_down:
    if ff[0] <= ff[1]:
        down_flag.append(ff[1])
    else:
        down_flag.append(0)

for ff in ff_left:
    if ff[0] <= ff[1]:
        left_flag.append(ff[1])
    else:
        left_flag.append(0)

# The second max layer
up_state.append(up_flag)
up_state.append(right_flag)
up_state.append(down_flag)
up_state.append(left_flag)
up_state = np.array(up_state, dtype = float)
up_state = up_state.T

up_count = 0
for st in up_state:
    if st[1] == np.max(st):
        up_count += 1
    else:
        pass
acc2 = up_count

'''
######################################################
#                   TEST DOWN STATE                  #
######################################################
'''
up_state = []
# Test the first max layer
up_flag = []
right_flag = []
down_flag = []
left_flag = []

''' Test '''
init.input_temp, init.output_temp = feature_extraction(3, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
test_up = init.input_temp
test_up = np.divide(test_up, 500.0)

ff_up = NN_UP.foward(test_up)
ff_right = NN_RIGHT.foward(test_up)
ff_down = NN_DOWN.foward(test_up)
ff_left = NN_LEFT.foward(test_up)

# The first max layer
for ff in ff_up:
    if ff[0] <= ff[1]:
        up_flag.append(ff[1])
    else:
        up_flag.append(0)

for ff in ff_right:
    if ff[0] <= ff[1]:
        right_flag.append(ff[1])
    else:
        right_flag.append(0)

for ff in ff_down:
    if ff[0] <= ff[1]:
        down_flag.append(ff[1])
    else:
        down_flag.append(0)

for ff in ff_left:
    if ff[0] <= ff[1]:
        left_flag.append(ff[1])
    else:
        left_flag.append(0)

# The second max layer
up_state.append(up_flag)
up_state.append(right_flag)
up_state.append(down_flag)
up_state.append(left_flag)
up_state = np.array(up_state, dtype = float)
up_state = up_state.T

up_count = 0
for st in up_state:
    if st[2] == np.max(st):
        up_count += 1
    else:
        pass
acc3 = up_count

'''
######################################################
#                   TEST LEFT STATE                  #
######################################################
'''
up_state = []
# Test the first max layer
up_flag = []
right_flag = []
down_flag = []
left_flag = []

''' Test '''
init.input_temp, init.output_temp = feature_extraction(4, 16, 1)    
init.input_temp = np.array(init.input_temp, dtype = float)
test_up = init.input_temp
test_up = np.divide(test_up, 500.0)

ff_up = NN_UP.foward(test_up)
ff_right = NN_RIGHT.foward(test_up)
ff_down = NN_DOWN.foward(test_up)
ff_left = NN_LEFT.foward(test_up)

# The first max layer
for ff in ff_up:
    if ff[0] <= ff[1]:
        up_flag.append(ff[1])
    else:
        up_flag.append(0)

for ff in ff_right:
    if ff[0] <= ff[1]:
        right_flag.append(ff[1])
    else:
        right_flag.append(0)

for ff in ff_down:
    if ff[0] <= ff[1]:
        down_flag.append(ff[1])
    else:
        down_flag.append(0)

for ff in ff_left:
    if ff[0] <= ff[1]:
        left_flag.append(ff[1])
    else:
        left_flag.append(0)

# The second max layer
up_state.append(up_flag)
up_state.append(right_flag)
up_state.append(down_flag)
up_state.append(left_flag)
up_state = np.array(up_state, dtype = float)
up_state = up_state.T

up_count = 0
for st in up_state:
    if st[3] == np.max(st):
        up_count += 1
    else:
        pass
acc4 = up_count

''' ############################################### '''
print up_state
print

print  '************************************'
print 'UP'
print init.UP_INPUT
print '************************************'
print 'RIGHT'
print init.RIGHT_INPUT
print '************************************'
print 'DOWN'
print init.DOWN_INPUT
print '************************************'
print 'LEFT'
print init.LEFT_INPUT
print '************************************'

print 'Accuracy of UP movement'
acc1 = acc1*100.0/65.0
print acc1
print

print 'Accuracy of RIGHT movement'
acc2 = acc2*100.0/65.0
print acc2
print

print 'Accuracy of DOWN movement'
acc3 = acc3*100.0/65.0
print acc3
print

print 'Accuracy of LEFT movement'
acc4 = acc4*100.0/65.0
print acc4
print

'''
print
print init.UP_INPUT
print 
print init.RIGHT_INPUT
print
print init.DOWN_INPUT
print
print init.LEFT_INPUT
'''


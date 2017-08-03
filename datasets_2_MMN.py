# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 05:30:05 2016
@name: datasets_2.py
@description: 
    - Create datasets for training 
    - Acquire signals and extract features online for using
@author: VPi
"""

''' -------------------------------------------------------------------------
     INITIALIZE VARIABLES FOR ACQUIRE SIGNALS BY USING API LIBRARIES
------------------------------------------------------------------------- '''
#import sys,os
import time
import ctypes

import numpy as np
from scipy import signal
from scipy.fftpack import fft

from ctypes import cdll
#from ctypes import CDLL
from ctypes import c_int
from ctypes import c_uint
from ctypes import pointer
#from ctypes import c_char_p
from ctypes import c_float
from ctypes import c_double
from ctypes import byref

libEDK = cdll.LoadLibrary("edk.dll")

ED_AF3=3
ED_F7=4
ED_F3=5
ED_FC5=6
ED_T7=7
ED_P7=8
ED_O1=9
ED_O2=10
ED_P8=11
ED_T8=12
ED_FC6=13
ED_F4=14
ED_F8=15
ED_AF4=16

targetChannelList = [ED_AF3, ED_F7, ED_F3, ED_FC5, ED_T7,ED_P7, ED_O1, ED_O2, ED_P8, ED_T8,ED_FC6, ED_F4, ED_F8, ED_AF4]
header = ['AF3','F7','F3', 'FC5', 'T7', 'P7', 'O1', 'O2','P8', 'T8', 'FC6', 'F4','F8', 'AF4']
#write = sys.stdout.write
eEvent      = libEDK.EE_EmoEngineEventCreate()
eState      = libEDK.EE_EmoStateCreate()
userID            = c_uint(0)
nSamples   = c_uint(0)
nSam       = c_uint(0)
nSamplesTaken  = pointer(nSamples)
#da = zeros(128,double)
data     = pointer(c_double(0))
user                    = pointer(userID)
composerPort          = c_uint(1726)
secs      = c_float(1)
datarate    = c_uint(0)
readytocollect    = False
option      = c_int(0)
state     = c_int(0)

print libEDK.EE_EngineConnect("Emotiv Systems-5")
if libEDK.EE_EngineConnect("Emotiv Systems-5") != 0:
    print "Emotiv Engine start up failed."

hData = libEDK.EE_DataCreate()
libEDK.EE_DataSetBufferSizeInSec(secs)

''' 
Function: acquire_data(ACQ_SIGNAL)
Description: acquire data from EPOC device by using API libraries
'''
def acquire_data():
    global state
    global readytocollect
    acquire_signal = []
    temp = []
    #count = 0
    #k = 0
    
    print "START COLLECTING SIGNAL"
    while (1):
        state = libEDK.EE_EngineGetNextEvent(eEvent)
        if state == 0:
            eventType = libEDK.EE_EmoEngineEventGetType(eEvent)
            libEDK.EE_EmoEngineEventGetUserId(eEvent, user)
            if eventType == 16: #libEDK.EE_Event_enum.EE_UserAdded:
                print "User added"
                libEDK.EE_DataAcquisitionEnable(userID,True)
                readytocollect = True
    
        if readytocollect==True:    
            libEDK.EE_DataUpdateHandle(0, hData)
            libEDK.EE_DataGetNumberOfSample(hData,nSamplesTaken)
            #print "Updated :",nSamplesTaken[0]
            if nSamplesTaken[0] != 0:
                nSam=nSamplesTaken[0]
                arr=(ctypes.c_double*nSamplesTaken[0])()
                ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
                #libEDK.EE_DataGet(hData, 3,byref(arr), nSam)                         
                #data = array('d')#zeros(nSamplesTaken[0],double)
                for sampleIdx in range(nSamplesTaken[0]): 
                    for i in range(14): 
                        libEDK.EE_DataGet(hData,targetChannelList[i],byref(arr), nSam)
                        temp.append(arr[sampleIdx])
                        #print temp
                    acquire_signal.append(temp)
                    #print acquire_signal
                    #print len(acquire_signal)
                    #print
                    temp = []
        #time.sleep(0.01)
        if len(acquire_signal) == 1280:
            break
        
    return acquire_signal

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
                
    # Energy of beta rhythm
    elif rhythm==2:
        for m in range(0,int(N/2)):
            if freq[m]>=14 and freq[m]<=30:
                E = E + WINDOW_SIGNAL[m]**2
                
    # Energy of whole of signal
    elif rhythm==3:
        for m in range(0,int(N/2)):
            E = E + WINDOW_SIGNAL[m]**2
            
    
    return E

''' -------------------------------------------------------------------------
         ACQUIRE SIGNALS AND EXTRACT FEATURES ONLINE FOR TRAINING
------------------------------------------------------------------------- '''

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
def feature_extraction(state):
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
    ACQ_SIGNAL = acquire_data()
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
        # Neutral state
        elif state==5:
            OUTPUT_DATASET.append([1,0])            
        
        # Update step i
        i = i + 16
    
    return INPUT_DATASET, OUTPUT_DATASET

''' -------------------------------------------------------------------------
         ACQUIRE SIGNALS AND EXTRACT FEATURES ONLINE FOR USING
------------------------------------------------------------------------- '''

''' 
Function: online_signal(flag)
Description: Acquiring online signals from EPOC by using API libraries
'''
def online_signal(acquire_online_signal):
    global state
    global readytocollect
    #acquire_online_signal = []
    temp_online = []
    #count = 0
    #k = 0
    
    print "START COLLECTING SIGNAL"

    state = libEDK.EE_EngineGetNextEvent(eEvent)
    if state == 0:
        eventType = libEDK.EE_EmoEngineEventGetType(eEvent)
        libEDK.EE_EmoEngineEventGetUserId(eEvent, user)
        if eventType == 16: #libEDK.EE_Event_enum.EE_UserAdded:
            print "User added"
            libEDK.EE_DataAcquisitionEnable(userID,True)
            readytocollect = True
    
    if readytocollect==True:    
        libEDK.EE_DataUpdateHandle(0, hData)
        libEDK.EE_DataGetNumberOfSample(hData,nSamplesTaken)
        #print "Updated :",nSamplesTaken[0]
        if nSamplesTaken[0] != 0:
            nSam=nSamplesTaken[0]
            arr=(ctypes.c_double*nSamplesTaken[0])()
            ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))
            #libEDK.EE_DataGet(hData, 3,byref(arr), nSam)                         
            #data = array('d')#zeros(nSamplesTaken[0],double)
            for sampleIdx in range(nSamplesTaken[0]): 
                for i in range(14): 
                    libEDK.EE_DataGet(hData,targetChannelList[i],byref(arr), nSam)
                    temp_online.append(arr[sampleIdx])
                    #print temp
                if len(acquire_online_signal) < 256:
                    acquire_online_signal.append(temp_online)
                else:
                    del acquire_online_signal[0]
                    acquire_online_signal.append(temp_online)
                #print acquire_signal
                #print len(acquire_signal)
                #print
                temp_online = []
    #time.sleep(0.01)
                
    return acquire_online_signal

def online_features_extraction(window):
    # reference parameter
    buffer_features = []
    electrode = []
    temp_features = 0
    
    # Initialize sampling rate and sampling frequency
    f = 128.0 # Sampling rate
    T = 1.0/128.0 # Sampling time
    t = np.linspace(1.0, 256.0, 256) # Sampling time
    t = np.divide(t, f)
    # Number of datas
    N = 256.0
    # Loop variables    
    j = 0
    # Acquire signals by calling acquire_data() 
    window = np.array(window, dtype = float)
    
    print 'START PROCESSING'
    
    for j in range(0, 14):
        # Get rectangle window from 'window' for each electrode
        electrode = window.T[j]
        # Apply IIR filter to get frequency from 8Hz to 30Hz
        electrode = IIR_filter(electrode)
        # Apply hanning window
        electrode = hanning_window(electrode)
        
        # Apply FFT to get spectrum
        electrode = fft(electrode)
        electrode = 2.0/N*np.abs(electrode[0:int(N/2)])
        freq = np.linspace(0.0, 1.0/(2*T), int(N/2))
        
        # Calculate energy alpha and store into temp_feature before storing into BUFFER_FEATURES
        temp_features = energy_calculation(1, N, electrode, freq)
        buffer_features.append(temp_features)
            
        # Calculate energy beta and store into temp_feature before storing into BUFFER_FEATURES
        temp_features = energy_calculation(2, N, electrode, freq)
        buffer_features.append(temp_features)
            
        # Calculate energy whole of signal and store into temp_feature before storing into BUFFER_FEATURES
        temp_features = energy_calculation(3, N, electrode, freq)
        buffer_features.append(temp_features)        
        
    return buffer_features

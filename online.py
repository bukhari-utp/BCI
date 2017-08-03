# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:06:16 2016
@name: online.py
@description: 
    - Acquire signal from EPOC device by using API libraries
    - Extract features and store into BUFFER_FEATURES
    - Apply features to ANN to get predictive state
@author: VPi
"""

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

def acquire_online():
    global state
    global readytocollect
    acquire_signal = []
    temp = []
    k = 0
    
    while (k<1025):
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
            print "Updated :",nSamplesTaken[0]
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
                    temp = []
        #time.sleep(0.2)
        k = k + 1
        
    return acquire_signal

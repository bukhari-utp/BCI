# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 16:33:14 2016
@name: initialization.py 
@description: Initialize global variables to use in application
@author: VPi
"""

import numpy as np

''' -------------------------------------------------------------------------
                       TRAINING AND ONLINE VARIABLES
------------------------------------------------------------------------- '''

''' Dataset input and output variables to apply to Artificial Neural Nets '''
# Dimension is 4100x42 if window is 2s and 4608x42 if window is 1s
# 42 is dimension of input including features: energy alpha, beta, and whole of signal for each electrode
# 4100 (each state has 1025 data) and 4608 (each state has 1152 data) is the number of window of 2s and 1s, respectively
INPUT_DATASETs = [] 
INPUT_DATASETs = np.array(INPUT_DATASETs, dtype = float)
# Dimension 1025x42
input_temp = []
# Dimension is 4100x4 if window is 2s and 4608x4 if window is 1s
# 4 is dimension of output correspond to 4 state: up, down, right, and left
OUTPUT_DATASETs = []
OUTPUT_DATASETs = np.array(OUTPUT_DATASETs, dtype = float)
# Dimension 1025x4
output_temp = []

''' Dataset variable for Modular Multi-layer Nets '''
UP_INPUT = np.array([], dtype = float)
RIGHT_INPUT = np.array([], dtype = float)
DOWN_INPUT = np.array([], dtype = float)
LEFT_INPUT = np.array([], dtype = float)

UP_OUTPUT = np.array([], dtype = float)
RIGHT_OUTPUT = np.array([], dtype = float)
DOWN_OUTPUT = np.array([], dtype = float)
LEFT_OUTPUT = np.array([], dtype = float)

''' Acquiring variable to acquire signals from EPOC device '''
''' In training process '''
# Dimension is 1280x14
# 14 is the number of electrode on EPOC device
# 1280 is the number of data during 10s
''' In online process '''
# Dimension is 256x14 if window is 2s and 128x14 if window is 1s
ACQ_SIGNAL = []

''' Window variable to create window and analyze spectrum '''
# Dimension is 256x14 if window is 2s and 128x14 if window is 1s
# Window variable is applied hanning window before performing FFT and extracting features
WINDOW_SIGNAL = []

''' Feature variables to save features before saving to INPUT_DATASET '''
# temp_feature variable can be energy alpha, beta or whole of signal
# It is saved FEATURES consequently
temp_features = 0
''' In training process '''
# Dimension is 1x24
# 24 is dimension of input including features: energy alpha, beta, and whole of signal for each electrode
# After collecting 24 feature, it is saved to INPUT_DATASET
''' In online process '''
# Dimension is 1x24 and it is applied directly to Artificial Neural Nets
BUFFER_FEATURES = []


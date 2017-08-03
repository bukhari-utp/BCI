# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:34:40 2016
@name: test_1.py
@description: Test performance of initialization.py and datasets.py
@author: VPi
"""

import numpy as np
import initialization as init
import datasets

press = raw_input('Enter 1 to begin collecting signal: ')

init.input_temp, init.output_temp = datasets.feature_extraction(1)

init.input_temp = np.array(init.input_temp, dtype = float)
init.output_temp = np.array(init.output_temp, dtype = float)

print init.input_temp.shape
print init.output_temp.shape

print init.input_temp

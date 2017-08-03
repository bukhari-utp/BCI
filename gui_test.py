# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 05:44:57 2016
@name: gui_test.py
@description: 
    - Create demo gui with 2 button
    - BEGIN button is used to start collect online signal and online extraction
    - STOP button is used to change flag
@author: VPi
"""

import initialization as init
import datasets_2 as dat2

import numpy as np
import sys, time
from PyQt4 import QtGui, QtCore
import threading

flag = 1
signal = []

def online():
    #init.ACQ_SIGNAL = dat2.online_signal(flag)
    global flag
    #SIGNAL = []
        
    print flag
    
    while flag==1:
        init.ACQ_SIGNAL = dat2.online_signal(init.ACQ_SIGNAL)
        
        print len(init.ACQ_SIGNAL)
        #SIGNAL = np.array(init.ACQ_SIGNAL)
        #print SIGNAL
        print
        print
        print
        print threading.active_count()
        print
        print 
        print
        time.sleep(1)

def online_2():
    global flag
    
    init.BUFFER_FEATURES = []
    init.WINDOW_SIGNAL = []
        
    while flag==1:
        if len(init.ACQ_SIGNAL)==256:
            init.WINDOW_SIGNAL = init.ACQ_SIGNAL
            init.BUFFER_FEATURES = dat2.online_features_extraction(init.WINDOW_SIGNAL)
        
            print init.BUFFER_FEATURES
            print len(init.BUFFER_FEATURES)
        else:
            pass
        
        time.sleep(1)

def stop():
    global flag
    flag = 0
    
    print flag

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        ''' Initialize basic feature '''
        self.setGeometry(100, 100, 200, 200) # Initialize dimension for GUI
        self.setWindowTitle("Brain - Computer Interface v.1.0") # Set title for GUI
        self.setWindowIcon(QtGui.QIcon('taegaryen.ico')) # Set icon
        
        self.home()
    
    def home(self):
        btn_t1 = QtGui.QPushButton('BEGIN', self)
        btn_t1.setGeometry(50, 20, 60, 30)
        btn_t1.clicked.connect(self.begin_online)
        
        btn_t2 = QtGui.QPushButton('STOP', self)
        btn_t2.setGeometry(50, 100, 60, 30)
        btn_t2.clicked.connect(self.stop_online)
    
    def begin_online(self):
        global flag
        
        if flag==0:
            flag = 1
            
        online_thread = threading.Thread(target=online)
        online_thread.start()
        
        online_thread_2 = threading.Thread(target=online_2)
        online_thread_2.start()
        
    def stop_online(self):
        stop_thread = threading.Thread(target=stop)
        stop_thread.start()

def run():
    app = QtGui.QApplication(sys.argv)
    # mac, plastique, sgi, windows
    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('mac'))
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

run()

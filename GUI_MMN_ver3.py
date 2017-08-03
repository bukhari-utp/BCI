# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 16:31:29 2016
@name: GUI_MMN_ver3.py
@description: Graphical user interface of BCI system
@date: Dec - 15 - 2016
@author: VPi
"""

# Import module
import initialization as init
import datasets_2_MMN as dat2
import ANN_MMN as ann

# Import libraries
import sys # To ensure application will nice, clean, close when exiting
#import os
import threading
import time
import serial
import numpy as np
from PyQt4 import QtGui, QtCore
import matplotlib.pyplot as plt

''' GUI variables '''
blink_up = 0
blink_right = 0
blink_down = 0
blink_left = 0

class Window(QtGui.QMainWindow):
    # Create variables to emit signal and display on progress bar
    value_progbar_1 = QtCore.pyqtSignal(int)
    value_progbar_2 = QtCore.pyqtSignal(int)
    value_progbar_3 = QtCore.pyqtSignal(int)
    value_progbar_4 = QtCore.pyqtSignal(int)
    value_progbar_5 = QtCore.pyqtSignal(int)
    
    # Update stimulus
    image_signal_1 = QtCore.pyqtSignal(int)
    image_signal_2 = QtCore.pyqtSignal(int)
    image_signal_3 = QtCore.pyqtSignal(int)
    image_signal_4 = QtCore.pyqtSignal(int)
    
    def __init__(self):
        super(Window, self).__init__()
        ''' Initialize basic feature '''
        self.setGeometry(100, 100, 1150, 500) # Initialize dimension for GUI
        self.setWindowTitle("Brain - Computer Interface v.1.0") # Set title for GUI
        self.setWindowIcon(QtGui.QIcon('taegaryen.ico')) # Set icon
        
        ''' Make tab '''
        self.tabs = QtGui.QTabWidget(self)        
        self.tabs.resize(1150, 480)
        
        # flag for connection
        self.flag_connection = 0
        
        # sensitive variables
        self.slider_up = 0
        self.slider_right = 0
        self.slider_down = 0
        self.slider_left = 0
        
        # serial variables
        self.serial_port = ''
        self.serial_baudrate = ''
        self.ser = serial.Serial()
        
        self.home()
    
    def home(self):        
        ''' Configure tab '''
        self.tab_train = QtGui.QWidget(self)
        self.tab_user = QtGui.QWidget(self)
        
        ''' ------------------------------------------------------------------
                                    Design tab Train
        ------------------------------------------------------------------ '''
        # Frame option
        frame_t1 = QtGui.QFrame(self.tab_train)
        frame_t1.setFrameShape(QtGui.QFrame.StyledPanel)
        frame_t1.move(5, 65)
        frame_t1.resize(600, 300)  
        # Label option
        label_t = QtGui.QLabel('Record Signals', self.tab_train)
        label_t.setGeometry(190, 0, 300, 60)
        label_t.setFont(QtGui.QFont('Times', 30))
        
        # Label for record signals
        label_t1 = QtGui.QLabel('Neutral', self.tab_train)
        label_t1.setGeometry(20, 90, 80, 50)
        label_t1.setFont(QtGui.QFont('Times', 15))
        
        label_t2 = QtGui.QLabel('Up', self.tab_train)
        label_t2.setGeometry(20, 140, 80, 50)
        label_t2.setFont(QtGui.QFont('Times', 15))
        
        label_t3 = QtGui.QLabel('Right', self.tab_train)
        label_t3.setGeometry(20, 190, 80, 50)
        label_t3.setFont(QtGui.QFont('Times', 15))
        
        label_t4 = QtGui.QLabel('Down', self.tab_train)
        label_t4.setGeometry(20, 240, 80, 50)
        label_t4.setFont(QtGui.QFont('Times', 15))
        
        label_t5 = QtGui.QLabel('Left', self.tab_train)
        label_t5.setGeometry(20, 290, 80, 50)
        label_t5.setFont(QtGui.QFont('Times', 15))
        
        # Progress bar
        self.progress_t1 = QtGui.QProgressBar(self.tab_train)
        self.progress_t1.setGeometry(120, 100, 400, 30)
        self.value_progbar_1.connect(self.progress_t1.setValue) # Emit progress and display on progress bar
        
        self.progress_t2 = QtGui.QProgressBar(self.tab_train)
        self.progress_t2.setGeometry(120, 150, 400, 30)
        self.value_progbar_2.connect(self.progress_t2.setValue)
        
        self.progress_t3 = QtGui.QProgressBar(self.tab_train)
        self.progress_t3.setGeometry(120, 200, 400, 30)
        self.value_progbar_3.connect(self.progress_t3.setValue)
        
        self.progress_t4 = QtGui.QProgressBar(self.tab_train)
        self.progress_t4.setGeometry(120, 250, 400, 30)
        self.value_progbar_4.connect(self.progress_t4.setValue)
        
        self.progress_t5 = QtGui.QProgressBar(self.tab_train)
        self.progress_t5.setGeometry(120, 300, 400, 30)
        self.value_progbar_5.connect(self.progress_t5.setValue)
        
        # Record button
        btn_t1 = QtGui.QPushButton('Record', self.tab_train)
        btn_t1.setGeometry(530, 100, 60, 30)
                
        btn_t2 = QtGui.QPushButton('Record', self.tab_train)
        btn_t2.setGeometry(530, 150, 60, 30)
                
        btn_t3 = QtGui.QPushButton('Record', self.tab_train)
        btn_t3.setGeometry(530, 200, 60, 30)
                
        btn_t4 = QtGui.QPushButton('Record', self.tab_train)
        btn_t4.setGeometry(530, 250, 60, 30)
                
        btn_t5 = QtGui.QPushButton('Record', self.tab_train)
        btn_t5.setGeometry(530, 300, 60, 30)
                
        btn_t1.clicked.connect(self.record_t1)
        btn_t2.clicked.connect(self.record_t2)
        btn_t3.clicked.connect(self.record_t3)
        btn_t4.clicked.connect(self.record_t4)
        btn_t5.clicked.connect(self.record_t5)
        
        # Train button
        btn_t = QtGui.QPushButton('Train', self.tab_train)
        btn_t.setGeometry(400, 390, 100, 50)
        btn_t.clicked.connect(self.training_ANN)
        
        # Reset button
        btn_trs = QtGui.QPushButton('Reset', self.tab_train)
        btn_trs.setGeometry(100, 390, 100, 50)
        btn_trs.clicked.connect(self.reset_button)
        
        ''' Arrow image patterns '''
        # Frame arrow
        
        frame_t2 = QtGui.QFrame(self.tab_train)
        frame_t2.setFrameShape(QtGui.QFrame.StyledPanel)
        frame_t2.move(650, 15)
        frame_t2.resize(470, 410)
        
        ''' Up arrow '''    
        self.label_img_t1 = QtGui.QLabel(self.tab_train)
        
        # Grey image
        self.img_t1 = QtGui.QPixmap('stimulus/up.png')
        self.img_t1 = self.img_t1.scaled(100, 150)        
        # Red image
        self.img_t11 = QtGui.QPixmap('stimulus/up1.png')
        self.img_t11 = self.img_t11.scaled(100, 150)
        
        self.label_img_t1.setPixmap(self.img_t1)
        self.label_img_t1.setGeometry(835, 0, 130, 200)        
        self.image_signal_1.connect(self.image1_process)
        
        ''' Right arrow '''
        self.label_img_t2 = QtGui.QLabel(self.tab_train)
        
        # Grey image
        self.img_t2 = QtGui.QPixmap('stimulus/right.png')
        self.img_t2 = self.img_t2.scaled(150, 100)
        # Red image
        self.img_t22 = QtGui.QPixmap('stimulus/right1.png')
        self.img_t22 = self.img_t22.scaled(150, 100)
        
        self.label_img_t2.setPixmap(self.img_t2)
        self.label_img_t2.setGeometry(960, 155, 200, 130)
        self.image_signal_2.connect(self.image2_process)
                
        ''' Down arrow '''
        self.label_img_t3 = QtGui.QLabel(self.tab_train)
        
        # Grey image
        self.img_t3 = QtGui.QPixmap('stimulus/down.png')
        self.img_t3 = self.img_t3.scaled(100, 150)
        #  Red image
        self.img_t33 = QtGui.QPixmap('stimulus/down1.png')
        self.img_t33 = self.img_t33.scaled(100, 150)
        
        self.label_img_t3.setPixmap(self.img_t3)
        self.label_img_t3.setGeometry(835, 240, 130, 200)
        self.image_signal_3.connect(self.image3_process)
                
        ''' Left arrow '''
        self.label_img_t4 = QtGui.QLabel(self.tab_train)
        
        # Grey image
        self.img_t4 = QtGui.QPixmap('stimulus/left.png')
        self.img_t4 = self.img_t4.scaled(150, 100)
        # Red image 
        self.img_t44 = QtGui.QPixmap('stimulus/left1.png')
        self.img_t44 = self.img_t44.scaled(150, 100)
        
        self.label_img_t4.setPixmap(self.img_t4)
        self.label_img_t4.setGeometry(660, 155, 200, 130)
        self.image_signal_4.connect(self.image4_process)

        
        ''' ------------------------------------------------------------------
                                    Design tab User 
        ------------------------------------------------------------------ '''
        ''' Arrow image patterns '''
        
        # Frame arrow
        frame_u1 = QtGui.QFrame(self.tab_user)
        frame_u1.setFrameShape(QtGui.QFrame.StyledPanel)
        frame_u1.move(50, 15)
        frame_u1.resize(470, 410)         
        
        ''' Up arrow '''                
        self.label_img_u1 = QtGui.QLabel(self.tab_user)
        self.label_img_u1.setGeometry(235, 0, 130, 200)
        self.label_img_u1.setPixmap(self.img_t1)
        
        ''' Right arrow '''
        self.label_img_u2 = QtGui.QLabel(self.tab_user)
        self.label_img_u2.setGeometry(360, 155, 200, 130)
        self.label_img_u2.setPixmap(self.img_t2)
        
        ''' Down arrow '''
        self.label_img_u3 = QtGui.QLabel(self.tab_user)
        self.label_img_u3.setGeometry(235, 240, 130, 200)
        self.label_img_u3.setPixmap(self.img_t3)
        
        ''' Left arrow '''
        self.label_img_u4 = QtGui.QLabel(self.tab_user)
        self.label_img_u4.setGeometry(60, 155, 200, 130)
        self.label_img_u4.setPixmap(self.img_t4)
        
        ''' Slider to change sensitiveness '''
        # Frame for sensitive slider
        frame_u2 = QtGui.QFrame(self.tab_user)
        frame_u2.setFrameShape(QtGui.QFrame.StyledPanel)
        frame_u2.move(600, 210)
        frame_u2.resize(500, 215)        
        
        # Up
        self.slider_u1 = QtGui.QSlider(QtCore.Qt.Horizontal, self.tab_user)
        self.slider_u1.setGeometry(680, 230, 400, 30)
        self.slider_u1.setRange(0, 10)
        self.slider_u1.setSliderPosition(5)
        
        # Right
        self.slider_u2 = QtGui.QSlider(QtCore.Qt.Horizontal, self.tab_user)
        self.slider_u2.setGeometry(680, 280, 400, 30)
        self.slider_u2.setRange(0, 10)
        self.slider_u2.setSliderPosition(5)
        
        # Down
        self.slider_u3 = QtGui.QSlider(QtCore.Qt.Horizontal, self.tab_user)
        self.slider_u3.setGeometry(680, 330, 400, 30)
        self.slider_u3.setRange(0, 10)
        self.slider_u3.setSliderPosition(5)
        
        # Left
        self.slider_u4 = QtGui.QSlider(QtCore.Qt.Horizontal, self.tab_user)
        self.slider_u4.setGeometry(680, 380, 400, 30)
        self.slider_u4.setRange(0, 10)
        self.slider_u4.setSliderPosition(5)
        
        # Function get slider value
        self.slider_u1.valueChanged.connect(self.slider_up_change_value)
        self.slider_u2.valueChanged.connect(self.slider_right_change_value)
        self.slider_u3.valueChanged.connect(self.slider_down_change_value)
        self.slider_u4.valueChanged.connect(self.slider_left_change_value)
        
        # Label for sensitive slider
        label_sens_u = QtGui.QLabel('Sensitive Adjustment', self.tab_user)
        label_sens_u.setGeometry(600, 150, 400, 60)
        label_sens_u.setFont(QtGui.QFont('Times', 30))     
        
        label_u1 = QtGui.QLabel('Up', self.tab_user)
        label_u1.setGeometry(620, 230, 200, 30)
        label_u1.setFont(QtGui.QFont('Times', 15))
        
        label_u2 = QtGui.QLabel('Right', self.tab_user)
        label_u2.setGeometry(620, 280, 200, 30)
        label_u2.setFont(QtGui.QFont('Times', 15))
        
        label_u3 = QtGui.QLabel('Down', self.tab_user)
        label_u3.setGeometry(620, 330, 200, 30)
        label_u3.setFont(QtGui.QFont('Times', 15))
        
        label_u4 = QtGui.QLabel('Left', self.tab_user)
        label_u4.setGeometry(620, 380, 200, 30)
        label_u4.setFont(QtGui.QFont('Times', 15))
        
        ''' Serial interface '''
        # Label for serial interface
        label_ser_u = QtGui.QLabel('Serial Configuration', self.tab_user)
        label_ser_u.setGeometry(600, 0, 400, 60)
        label_ser_u.setFont(QtGui.QFont('Times', 30))  
        
        # Frame for serial interface
        frame_u3 = QtGui.QFrame(self.tab_user)
        frame_u3.setFrameShape(QtGui.QFrame.StyledPanel)
        frame_u3.move(600, 60)
        frame_u3.resize(500, 90)
        
        # Label for serial configuration
        label_u5 = QtGui.QLabel('COM Port', self.tab_user)
        label_u5.setGeometry(620, 70, 200, 30)
        label_u5.setFont(QtGui.QFont('Times', 15))
        
        label_u6 = QtGui.QLabel('Baud rate', self.tab_user)
        label_u6.setGeometry(860, 70, 200, 30)
        label_u6.setFont(QtGui.QFont('Times', 15))
        
        # Combo box for serial configuration
        # COM port combobox
        self.combo_u1 = QtGui.QComboBox(self.tab_user)
        self.combo_u1.setGeometry(720, 70, 120, 30)
        self.combo_u1.addItem('COM24')
        self.combo_u1.addItem('COM25')
        self.combo_u1.addItem('COM6')
        self.combo_u1.addItem('COM11')
        self.combo_u1.addItem('COM3')
        self.combo_u1.addItem('COM4')
        self.combo_u1.addItem('COM5')
        self.combo_u1.addItem('COM31')
        
        # Baud rate combobox
        self.combo_u2 = QtGui.QComboBox(self.tab_user)
        self.combo_u2.setGeometry(950, 70, 120, 30)
        self.combo_u2.addItem('9600')
        self.combo_u2.addItem('19200')
        self.combo_u2.addItem('38400')
        self.combo_u2.addItem('57600')
        self.combo_u2.addItem('115200')
        
        # Get string from combo box
        self.combo_u1.activated[str].connect(self.change_port)
        self.combo_u2.activated[str].connect(self.change_baudrate)
        
        # Button connect
        self.btn_u1 = QtGui.QPushButton('Connect', self.tab_user)
        self.btn_u1.setGeometry(620, 110, 120, 30)
        self.btn_u1.clicked.connect(self.connect_button)
        
        ''' Add tabs to tabs object '''
        self.tabs.addTab(self.tab_train, "Train")
        self.tabs.addTab(self.tab_user, "User")
    
        self.show()
    
    ''' Record function '''
    ''' COLLECT NEUTRAL SIGNALS '''
    def record_t1(self):
        print 'COLLECTING NEUTRAL SIGNALS'
        thread_record_1 = threading.Thread(target=self.record_train_1)
        thread_record_1.start()
        thread_display_1 = threading.Thread(target=self.display_progbar_1)
        thread_display_1.start()
    
    # Thread collect data of NEUTRAL signals    
    def record_train_1(self):
        init.input_temp, init.output_temp = dat2.feature_extraction(5)
        init.input_temp = np.array(init.input_temp, dtype = float)
        init.output_temp = np.array(init.output_temp, dtype = float)
        
        init.UP_INPUT = init.input_temp
        init.RIGHT_INPUT = init.input_temp
        init.DOWN_INPUT = init.input_temp
        init.LEFT_INPUT = init.input_temp

        init.UP_OUTPUT = init.output_temp        
        init.RIGHT_OUTPUT = init.output_temp
        init.DOWN_OUTPUT = init.output_temp
        init.LEFT_OUTPUT = init.output_temp
        
        print
        print init.UP_INPUT.shape
        print init.RIGHT_INPUT.shape
        print init.DOWN_INPUT.shape
        print init.LEFT_INPUT.shape
        print
        
    # Display progress on progress bar
    def display_progbar_1(self):
        completed = 0
        
        while completed < 100:
            completed += 1
            self.value_progbar_1.emit(completed)
            time.sleep(0.1)    
                            
    ''' COLLECT UP SIGNALS '''
    def record_t2(self):
        print 'COLLECTING UP SIGNALS'
        thread_record_2 = threading.Thread(target=self.record_train_2)
        thread_record_2.start()
        thread_display_2 = threading.Thread(target=self.display_progbar_2)
        thread_display_2.start()
    
    # Thread collect data of UP signals    
    def record_train_2(self):
        init.input_temp, init.output_temp = dat2.feature_extraction(1)
        init.input_temp = np.array(init.input_temp, dtype = float)
        init.output_temp = np.array(init.output_temp, dtype = float)
        
        init.UP_INPUT = np.concatenate((init.UP_INPUT, init.input_temp), axis = 0)
        init.UP_OUTPUT = np.concatenate((init.UP_OUTPUT, init.output_temp), axis = 0)
        
        print init.UP_INPUT.shape
        print init.UP_INPUT
        print
        print init.UP_OUTPUT.shape
        print init.UP_OUTPUT
        print
        
    # Display progress on progress bar
    def display_progbar_2(self):
        global blink_up
        completed = 0
        
        blink_up = 1
        self.image_signal_1.emit(completed)
        while completed < 100:
            completed += 1
            self.value_progbar_2.emit(completed)
            time.sleep(0.1)
        
        blink_up = 0
        self.image_signal_1.emit(completed)
    
    def image1_process(self):
        global blink_up
        print 'blink up ' + str(blink_up)
        
        if blink_up == 1:
            # Active for tab train
            self.label_img_t1.clear()
            self.label_img_t1.setPixmap(self.img_t11)
            # Active for tab user
            self.label_img_u1.clear()
            self.label_img_u1.setPixmap(self.img_t11)
        else:
            # For tab train
            self.label_img_t1.clear()
            self.label_img_t1.setPixmap(self.img_t1)
            # For tab user
            self.label_img_u1.clear()
            self.label_img_u1.setPixmap(self.img_t1)
            
    ''' COLLECT RIGHT SIGNALS '''            
    def record_t3(self):
        print 'COLLECTING RIGHT SIGNALS'
        thread_record_3 = threading.Thread(target=self.record_train_3)
        thread_record_3.start()
        thread_display_3 = threading.Thread(target=self.display_progbar_3)
        thread_display_3.start()
    
    # Thread collect data of UP signals    
    def record_train_3(self):
        init.input_temp, init.output_temp = dat2.feature_extraction(2)
        init.input_temp = np.array(init.input_temp, dtype = float)
        init.output_temp = np.array(init.output_temp, dtype = float)
        
        init.RIGHT_INPUT = np.concatenate((init.RIGHT_INPUT, init.input_temp), axis = 0)
        init.RIGHT_OUTPUT = np.concatenate((init.RIGHT_OUTPUT, init.output_temp), axis = 0)
        
        print init.RIGHT_INPUT.shape
        print init.RIGHT_INPUT
        print
        print init.RIGHT_OUTPUT.shape
        print init.RIGHT_OUTPUT
        print
        
    # Display progress on progress bar
    def display_progbar_3(self):
        global blink_right
        completed = 0
        
        blink_right = 1
        self.image_signal_2.emit(completed)
        while completed < 100:
            completed += 1
            self.value_progbar_3.emit(completed)
            time.sleep(0.1)
        
        blink_right = 0
        self.image_signal_2.emit(completed)
    
    def image2_process(self):
        global blink_right
        print 'blink right ' + str(blink_right)
        
        if blink_right == 1:
            # Active for tab train
            self.label_img_t2.clear()
            self.label_img_t2.setPixmap(self.img_t22)
            # Active for tab user
            self.label_img_u2.clear()
            self.label_img_u2.setPixmap(self.img_t22)
        else:
            # For tab train
            self.label_img_t2.clear()
            self.label_img_t2.setPixmap(self.img_t2)
            # For tab user
            self.label_img_u2.clear()
            self.label_img_u2.setPixmap(self.img_t2)
    
    ''' COLLECT DOWN SIGNALS '''
    def record_t4(self):
        print 'COLLECTING DOWN SIGNALS'
        thread_record_4 = threading.Thread(target=self.record_train_4)
        thread_record_4.start()
        thread_display_4 = threading.Thread(target=self.display_progbar_4)
        thread_display_4.start()
    
    # Thread collect data of UP signals    
    def record_train_4(self):
        init.input_temp, init.output_temp = dat2.feature_extraction(3)
        init.input_temp = np.array(init.input_temp, dtype = float)
        init.output_temp = np.array(init.output_temp, dtype = float)
        
        init.DOWN_INPUT = np.concatenate((init.DOWN_INPUT, init.input_temp), axis = 0)
        init.DOWN_OUTPUT = np.concatenate((init.DOWN_OUTPUT, init.output_temp), axis = 0)
        
        print init.DOWN_INPUT.shape
        print init.DOWN_INPUT
        print
        print init.DOWN_OUTPUT.shape
        print init.DOWN_OUTPUT
        print
        
    # Display progress on progress bar
    def display_progbar_4(self):
        global blink_down
        completed = 0
        
        blink_down = 1
        self.image_signal_3.emit(completed)
        while completed < 100:
            completed += 1
            self.value_progbar_4.emit(completed)
            time.sleep(0.1)
        blink_down = 0
        self.image_signal_3.emit(completed)
    
    def image3_process(self):
        global blink_down
        print 'blink_down ' + str(blink_down)
        
        if blink_down == 1:
            # Active for tab train
            self.label_img_t3.clear()
            self.label_img_t3.setPixmap(self.img_t33)
            # Active for tab user
            self.label_img_u3.clear()
            self.label_img_u3.setPixmap(self.img_t33)
        else:
            # For tab train
            self.label_img_t3.clear()
            self.label_img_t3.setPixmap(self.img_t3)
            # For tab user
            self.label_img_u3.clear()
            self.label_img_u3.setPixmap(self.img_t3)
        
    ''' COLLECT LEFT SIGNALS '''
    def record_t5(self):
        print 'COLLECTING LEFT SIGNALS'
        thread_record_5 = threading.Thread(target=self.record_train_5)
        thread_record_5.start()
        thread_display_5 = threading.Thread(target=self.display_progbar_5)
        thread_display_5.start()
    
    # Thread collect data of UP signals    
    def record_train_5(self):
        init.input_temp, init.output_temp = dat2.feature_extraction(4)
        init.input_temp = np.array(init.input_temp, dtype = float)
        init.output_temp = np.array(init.output_temp, dtype = float)
        
        init.LEFT_INPUT = np.concatenate((init.LEFT_INPUT, init.input_temp), axis = 0)
        init.LEFT_OUTPUT = np.concatenate((init.LEFT_OUTPUT, init.output_temp), axis = 0)
        
        print init.LEFT_INPUT.shape
        print init.LEFT_INPUT
        print
        print init.LEFT_OUTPUT.shape
        print init.LEFT_OUTPUT
        print
        
    # Display progress on progress bar
    def display_progbar_5(self):
        global blink_left
        completed = 0
        
        blink_left = 1
        self.image_signal_4.emit(completed)
        while completed < 100:
            completed += 1
            self.value_progbar_5.emit(completed)
            time.sleep(0.1)
        
        blink_left = 0
        self.image_signal_4.emit(completed)
    
    def image4_process(self):
        global blink_left
        print 'blink_left ' + str(blink_left)
        
        if blink_left == 1:
            # Active for tab train
            self.label_img_t4.clear()
            self.label_img_t4.setPixmap(self.img_t44)
            # Active for tab user
            self.label_img_u4.clear()
            self.label_img_u4.setPixmap(self.img_t44)
        else:
            # For tab train
            self.label_img_t4.clear()
            self.label_img_t4.setPixmap(self.img_t4)
            # For tab user
            self.label_img_u4.clear()
            self.label_img_u4.setPixmap(self.img_t4)
    
    ''' TRAINING ANN '''
    def training_ANN(self):
        # Normalize because collective signals have E so high
        init.UP_INPUT = np.divide(init.UP_INPUT, 500)
        init.RIGHT_INPUT = np.divide(init.RIGHT_INPUT, 500)
        init.DOWN_INPUT = np.divide(init.DOWN_INPUT, 500)
        init.LEFT_INPUT = np.divide(init.LEFT_INPUT, 500)
        
        # UP Neural Nets
        self.NN_UP = ann.Neural_Network(Lambda = 0.0005)
        self.T_UP = ann.trainer(self.NN_UP)
        self.T_UP.train(init.UP_INPUT, init.UP_OUTPUT)
        
        # RIGHT Neural Nets
        self.NN_RIGHT = ann.Neural_Network(Lambda = 0.0005)
        self.T_RIGHT = ann.trainer(self.NN_RIGHT)
        self.T_RIGHT.train(init.RIGHT_INPUT, init.RIGHT_OUTPUT)
        
        # DOWN Neural Nets
        self.NN_DOWN = ann.Neural_Network(Lambda = 0.0005)
        self.T_DOWN = ann.trainer(self.NN_DOWN)
        self.T_DOWN.train(init.DOWN_INPUT, init.DOWN_OUTPUT)
        
        # LEFT Neural Nets
        self.NN_LEFT = ann.Neural_Network(Lambda = 0.0005)
        self.T_LEFT = ann.trainer(self.NN_LEFT)
        self.T_LEFT.train(init.LEFT_INPUT, init.LEFT_OUTPUT)
        
        ''' Draw training data, relation between T and error E '''
        #plt.figure(1)
        
        # Up training line
        plt.subplot(221)
        plt.plot(self.T_UP.E)
        plt.title('Training line of UP state')
        plt.grid(1)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        
        # Right training line
        plt.subplot(223)
        plt.plot(self.T_RIGHT.E)
        #plt.title('Training line of RIGHT state')
        plt.grid(1)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        
        # Down training line
        plt.subplot(222)
        plt.plot(self.T_DOWN.E)
        plt.title('Training line of DOWN state')
        plt.grid(1)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        
        # Left training line
        plt.subplot(224)
        plt.plot(self.T_LEFT.E)
        plt.title('Training line of LEFT state')
        plt.grid(1)
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
                
        plt.show()
    
    ''' RESET BUTTON '''
    def reset_button(self):
        init.UP_INPUT = np.array([], dtype = float)
        init.RIGHT_INPUT = np.array([], dtype = float)
        init.DOWN_INPUT = np.array([], dtype = float)
        init.LEFT_INPUT = np.array([], dtype = float)
        
        init.UP_OUTPUT = np.array([], dtype = float)
        init.RIGHT_OUTPUT = np.array([], dtype = float)
        init.DOWN_OUTPUT = np.array([], dtype = float)
        init.LEFT_OUTPUT = np.array([], dtype = float)
        
        print init.UP_INPUT
        print
        self.progress_t1.setValue(0)
        self.progress_t2.setValue(0)
        self.progress_t3.setValue(0)
        self.progress_t4.setValue(0)
        self.progress_t5.setValue(0)
    
    ''' CONNECT BUTTON '''
    def connect_button(self):
        # Switch state of connect button
        if self.flag_connection == 0:
            self.flag_connection = 1
            self.btn_u1.setText('Disconnect')
        elif self.flag_connection == 1:
            self.flag_connection = 0
            self.btn_u1.setText('Connect')
        
        # Check connection
        if self.flag_connection==1:
            print
            print 'WE ARE CONNECTING'
            print
            self.ser = serial.Serial(self.serial_port, self.serial_baudrate)
            print 
            print self.ser.isOpen()
            print
            thread_acquire_online = threading.Thread(target=self.acquire_online)
            thread_acquire_online.start()
            thread_extract_online = threading.Thread(target=self.extract_online)
            thread_extract_online.start()
        elif self.flag_connection==0:
            print
            print 'WE WERE DISCONNECT'
            print
            time.sleep(0.5)
            self.ser.close()
            print
            print self.ser.isOpen()
            print
    
    # Acquire online signal
    def acquire_online(self):
        while self.flag_connection==1:
            init.ACQ_SIGNAL = dat2.online_signal(init.ACQ_SIGNAL)
            print 'Length of init.ACQ_SIGNAL is: ' + str(len(init.ACQ_SIGNAL))
            print
            time.sleep(1)
    
    # Extract online signal
    def extract_online(self):
        global blink_up
        global blink_right
        global blink_down
        global blink_left
        
        feedforward_up = np.array([], dtype = float)
        feedforward_right = np.array([], dtype = float)
        feedforward_down = np.array([], dtype = float)
        feedforward_left = np.array([], dtype = float)
        
        up_flag = 0.0
        right_flag = 0.0
        down_flag = 0.0
        left_flag = 0.0
        
        state_flag = []
        
        init.WINDOW_SIGNAL = []
        init.BUFFER_FEATURES = []
        
        while self.flag_connection==1:
            if len(init.ACQ_SIGNAL)==256:
                state_flag = []
                init.WINDOW_SIGNAL = init.ACQ_SIGNAL
                init.BUFFER_FEATURES = dat2.online_features_extraction(init.WINDOW_SIGNAL)
                
                # Normalize input
                init.BUFFER_FEATURES = np.divide(init.BUFFER_FEATURES, 500)
                
                # Apply to ANN classification
                feedforward_up = self.NN_UP.foward(init.BUFFER_FEATURES)
                feedforward_right = self.NN_RIGHT.foward(init.BUFFER_FEATURES)
                feedforward_down = self.NN_DOWN.foward(init.BUFFER_FEATURES)
                feedforward_left = self.NN_LEFT.foward(init.BUFFER_FEATURES)
                
                # Check UP ANN
                if feedforward_up[0] <= feedforward_up[1]:
                    up_flag = np.max(feedforward_up)
                else:
                    up_flag = 0.0
                
                # Check RIGHT ANN
                if feedforward_right[0] <= feedforward_right[1]:
                    right_flag = np.max(feedforward_right)
                else:
                    right_flag = 0.0
                
                # Check DOWN ANN
                if feedforward_down[0] <= feedforward_down[1]:
                    down_flag = np.max(feedforward_down)
                else:
                    down_flag = 0.0
                
                # Check LEFT ANN
                if feedforward_left[0] <= feedforward_left[1]:
                    left_flag = np.max(feedforward_left)
                else:
                    left_flag = 0.0
                    
                state_flag.append(up_flag)
                state_flag.append(right_flag)
                state_flag.append(down_flag)
                state_flag.append(left_flag)
                
                state_flag = np.array(state_flag, dtype = float)
                                  
                if up_flag==0 and right_flag==0 and down_flag==0 and left_flag==0:
                    print
                    print 'ROBOT RELAX'
                    print
                    # Collect signal to return arrow
                    blink_up = 0
                    blink_right = 0
                    blink_down = 0
                    blink_left = 0
                    self.image_signal_1.emit(1)
                    self.image_signal_2.emit(1)
                    self.image_signal_3.emit(1)
                    self.image_signal_4.emit(1)
                    self.ser.write('n')
                elif up_flag!=0 or right_flag!=0 or down_flag!=0 or left_flag!=0:
                    print 'CHOOSE STATE'
                    # Check state
                    if state_flag[0]==np.max(state_flag):
                        print
                        print 'ROBOT MOVE STRAINGHT'
                        print
                        # Collect signal to active arrow
                        blink_up = 1
                        self.image_signal_1.emit(1)
                        self.ser.write('u')
                    elif state_flag[1]==np.max(state_flag):
                        print
                        print 'ROBOT TURN RIGHT'
                        print
                        # Collect signal to active arrow
                        blink_right = 1
                        self.image_signal_2.emit(1)
                        self.ser.write('r')
                    elif state_flag[2]==np.max(state_flag):
                        print
                        print 'ROBOT MOVE BACK'
                        print
                        # Collect signal to active arrow
                        blink_down = 1
                        self.image_signal_3.emit(1)
                        self.ser.write('d')
                    elif state_flag[3]==np.max(state_flag):
                        print
                        print 'ROBOT TURN LEFT' 
                        print
                        # Collect signal to active arrow
                        blink_left = 1
                        self.image_signal_4.emit(1)
                        self.ser.write('l')
                    
                time.sleep(0.25)
            else:
                pass
    
    ''' Sensitive function '''
    # Slider up
    def slider_up_change_value(self):
        self.slider_up = self.slider_u1.value()/50.0 - 0.1
        print self.slider_up
    
    # Slider right
    def slider_right_change_value(self):
        self.slider_right = self.slider_u2.value()/50.0 - 0.1
        print self.slider_right
    
    # Slider down
    def slider_down_change_value(self):
        self.slider_down = self.slider_u3.value()/50.0 - 0.1
        print self.slider_down
    
    # Slider left
    def slider_left_change_value(self):
        self.slider_left = self.slider_u4.value()/50.0 - 0.1
        print self.slider_left
    
    ''' Serial value '''
    def change_port(self, text):
        self.serial_port = str(text)
        print self.serial_port
    
    def change_baudrate(self, text):
        self.serial_baudrate = int(text)
        print self.serial_baudrate

#class ThreadWorker1()

def run():
    app = QtGui.QApplication(sys.argv)
    # mac, plastique, sgi, windows
    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('mac'))
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

run()

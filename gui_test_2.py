# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 09:01:22 2016
@name: gui_test_2.py
@description: Divide GUI to threads
@author: VPi
"""

import numpy as np
import sys, time
from PyQt4 import QtGui, QtCore
import threading

class ThreadWorker(QtCore.QThread):
    def __init__(self, obj):
        QtCore.QThread.__init__(self)
        #threading.Thread.__init__(self)
        self.obj = obj

    def gif(self):
        # set up the movie screen on a label
        self.movie_screen = QtGui.QLabel(self.obj)
        # expand and center the label 
        self.movie_screen.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)    
        #self.movie_screen.setAlignment(QtCore.Qt.AlignCenter)
        self.movie_screen.setGeometry(300, 20, 130, 200)
                
         # use an animated gif file you have in the working folder
        # or give the full file path
        self.movie = QtGui.QMovie("stimulus/down_0_5Hz.gif", QtCore.QByteArray(), self.obj) 
        self.movie.setCacheMode(QtGui.QMovie.CacheAll) 
        self.movie.scaledSize()
        self.movie.setSpeed(100) 
        self.movie_screen.setMovie(self.movie) 
        self.movie.start()
    
    def run(self):
        self.gif()

class Window(QtGui.QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        ''' Initialize basic feature '''
        self.setGeometry(700, 100, 500, 200) # Initialize dimension for GUI
        self.setWindowTitle("Brain - Computer Interface v.1.0") # Set title for GUI
        self.setWindowIcon(QtGui.QIcon('taegaryen.ico')) # Set icon
        
        self.animation = ThreadWorker(self)
        self.animation.start()
        
        self.home()
    
    def home(self):
        btn_t1 = QtGui.QPushButton('BEGIN', self)
        btn_t1.setGeometry(50, 20, 60, 30)
        btn_t1.clicked.connect(self.begin_online)
        
        btn_t2 = QtGui.QPushButton('STOP', self)
        btn_t2.setGeometry(50, 100, 60, 30)
        #btn_t2.clicked.connect(self.stop_online)
    
    def begin_online(self):
        a = 0
        
        while(a<10):
            print a
            time.sleep(1)
            a+=1

def run():
    app = QtGui.QApplication(sys.argv)
    # mac, plastique, sgi, windows
    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('mac'))
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

run()


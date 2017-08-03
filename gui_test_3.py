# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 10:38:55 2016
@name: gui_test_3.py
@description: Test ability of signal-slot mechanism in GUI
@author: VPi
"""

import sys
import threading
from PyQt4 import QtCore, QtGui

class Window(QtGui.QMainWindow):
    value_update = QtCore.pyqtSignal(int)
    
    def __init__(self):
        super(Window, self).__init__()
        ''' Initialize basic feature '''
        self.setGeometry(700, 100, 500, 200) # Initialize dimension for GUI
        self.setWindowTitle("Brain - Computer Interface v.1.0") # Set title for GUI
        self.setWindowIcon(QtGui.QIcon('taegaryen.ico')) # Set icon
        
        # Update progress bar variable
        #self.value_update = QtCore.pyqtSignal(int)
        
        self.statusBar()
        
        self.progress_b = QtGui.QProgressBar(self)
        self.progress_b.setGeometry(0, 0, 200, 30)
        #self.progress_b.valueChanged.connect(self.value_update)        
        self.value_update.connect(self.progress_b.setValue)
        
        self.btn = QtGui.QPushButton('ACTIVE', self)
        self.btn.setGeometry(0, 50, 50, 50)
        self.btn.clicked.connect(self.update_probar)
        
    def update_probar(self):
        sender = self.sender()
        self.statusBar().showMessage(sender.text() + ' was pressed')
        print 'BUTTON IS PRESSED'
        
        update_thread = threading.Thread(target=self.increase_number)
        update_thread.start()
    
    def increase_number(self):
        self.completed = 0
        
        while self.completed < 101:
            self.value_update.emit(self.completed)
            self.completed += 0.0001
        

def run():
    app = QtGui.QApplication(sys.argv)
    # mac, plastique, sgi, windows
    QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('mac'))
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())

run()
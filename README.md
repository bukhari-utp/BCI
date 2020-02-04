# BCI

There are 4 main files: 
1. GUI_MMN_ver3.py
2. datasets_2_MMN.py
3. initialization.py and
4. ANN_MMN.py.

Run:
python GUI_MMN_ver3.py

YT: https://www.youtube.com/watch?v=ARXPPh2w2VY

Paper: http://www.cs.cmu.edu/~tanja/BCI/BCIreview.pdf

Strategy to control the robot for motor-imagery-based system, I imagined that I was moving my arms to control different movements of the robot. 

The features were the power of the alpha/mu and beta (because this is the BCI system based on sensorimotor cortex).
 
TODO:
1. Save trained NN
2. Acquire: pre-task -- task -- post-task
 
Requirement:
  1. python=3.4
  2. scipy
  3. matplotlib
  4. PyQt4; sudo -H python3 -m pip install pyqt5
  5. pyserial

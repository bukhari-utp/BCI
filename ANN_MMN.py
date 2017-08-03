# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 16:58:07 2016
@name: ANN.py
@description: 
    - Initialize Artificial Neural Nets 
    - Create Training class to train ANN
@author: VPi
"""
import numpy as np

'''
----------------------------------------------------------------------
Design ANN
----------------------------------------------------------------------
'''

""" Class Feed - Foward ANN """
class Neural_Network(object):
	def __init__(self, Lambda = 0):
		# Define Hyperparameters
		self.inputLayerSize = 42
		self.outputLayerSize = 2
		self.hiddenLayerSize = 50
		
		# Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
		
		#Regularization Parameter:
		self.Lambda = Lambda
	
	def sigmoid(self, z):
		# Apply sigmoid activation function to scalar, vector or matrix
		return 1/(1+np.exp(-z))
	
	def sigmoidPrime(self, z):
		# Gradient of sigmoid (or derivative sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)
	
	def foward(self, X):
		# Propagate input through net
		self.sigma2 = np.dot(X, self.W1)
		self.o2 = self.sigmoid(self.sigma2)
		self.sigma3 = np.dot(self.o2, self.W2)
		yHat = self.sigmoid(self.sigma3)
		
		return yHat
	
	def costFunction(self, X, y):
		# Compute cost of y and yHat
		self.yHat = self.foward(X)
		
		# We don't want cost to increase with the number of examples,
		# so normalize by dividing the error term by number of examples(X.shape[0])
		E = 0.5*sum((y - self.yHat)**2)/X.shape[0] + (self.Lambda/2)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return E
	
	def costFunctionPrime(self, X, y):
		# Comput derivative with respect to W1 and W2 for a given X, y
		self.yHat = self.foward(X)
		
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.sigma3))
		#Add gradient of regularization term:
		dEdW2 = np.dot(self.o2.T, delta3)/X.shape[0] + self.Lambda*self.W2
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.sigma2)
		#Add gradient of regularization term:
		dEdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1
		
		return dEdW1, dEdW2
		
	''' Helper Functions for interacting with other classes '''
	def getParams(self):
		# Get W1 and W2 unrolled into vector
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params
	
	def setParams(self, params):
		# Set W1 and W2 using single parameter vector
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))
	
	def computeGradients(self, X, y):
		dEdW1, dEdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dEdW1.ravel(), dEdW2.ravel()))
	
	def computeNumericalGradient(N, X, y):
		paramsInitial = N.getParams()
		numgrad = np.zeros(paramsInitial.shape)
		perturb = np.zeros(paramsInitial.shape)
		e = 1e-4
		
		for p in range(len(paramsInitial)):
			#Set perturbation vector
			perturb[p] = e
			N.setParams(paramsInitial + perturb)
			loss2 = N.costFunction(X, y)
			
			N.setParams(paramsInitial - perturb)
			loss1 = N.costFunction(X, y)
			
			#Compute Numerical Gradient
			numgrad[p] = (loss2 - loss1) / (2*e)
			
			#Return the value we changed to zero:
			perturb[p] = 0
			
		#Return Params to original value:
		N.setParams(paramsInitial)
		
		return numgrad 

from scipy import optimize#, stats

class trainer(object):
	def __init__(self, N):
		# Make Local reference to net
		self.N = N
	
	def callbackF(self, params):
		self.N.setParams(params)
		self.E.append(self.N.costFunction(self.X, self.y))
	
	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X, y)
		
		return cost, grad
	
	def train(self, X, y):
		# Make an internal for the call back function
		self.X = X
		self.y = y
		
		# Make empty list to store costs:
		self.E = []
		
		params0 = self.N.getParams()
		
		options = {'maxiter': 500, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='L-BFGS-B', \
				args=(X, y), options=options, callback=self.callbackF)
		
		self.N.setParams(_res.x)
		self.optimizationResults = _res


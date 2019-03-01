# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:17:46 2019

@author: Varad Srivastava
"""
#Creating an artificial neuron
import numpy as np
import matplotlib.pyplot as plt

class arti_neuron:
        def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes,learning_rate):
            self.no_of_in_nodes = no_of_in_nodes
            self.no_of_out_nodes = no_of_out_nodes 
            self.no_of_hidden_nodes = no_of_hidden_nodes
        
        #self.weights=
        
    def func(x):
        return 1/(1+np.exp(-x))

    
    #derivative of our function(confidence that the weight)
    def func_derivative(self, x):
    return x*(1-x)
    
     #passing training set through our artificial neuron
    def train(self, train_inputs, train_outputs, iterations):
        for iteration in xrange(iterations):
            output = self.think(train_inputs)
            
            #checking for error
            error = train_outputs - output      
            
            
            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(train_inputs.T, error * self.func_derivative(output))

            # Adjust the weights.
            self.weights += adjustment
    
    #Our neuron thinks about the output        
    def think(self, inputs):
        #passing the input to neuron
        return self.func(dot(inputs, self.weights))
        
        
            
            
            
            
            
        

    
        

        
        
#main
inputs = np.array([[0,1],[1,0],[1,1],[0,0]])
outputs = np.array([0,1,1,0]).T    

    
plt.plot(x, func(x),'b')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Sigmoid Function')
plt.grid()
plt.show()


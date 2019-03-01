# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:44:49 2019

@author: Varad Srivastava
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


w = np.array([-0.5, 1, 1])
def OR(x1, x2):
    x = np.array([1, x1, x2])
    w = np.array([-0.5, 1, 1])
    y = np.sum(w*x)
    if y <= 0:
        return 0
    else:
        return 1
    
if __name__ == '__main__':
    input = [(0, 0), (1, 0), (0, 1), (1, 1)]
    print("OR")
    for x in input:
        y = OR(x[0], x[1])
        print(str(x) + " -> " + str(y)) 
        

      
#decision boundary
i = [0,1,0,1]
o = [0,0,1,1]
plt.style.use('ggplot')
X = -w[0] / w[1]
Y = -w[0] / w[2]
D = Y
C = -Y / X
line_x_coords = np.array([0, X])
line_y_coords = C * line_x_coords + D
plt.plot(line_x_coords, line_y_coords)
plt.scatter(i,o, color='red',s=100)  
plt.scatter(0,0,color='blue',s=100,marker="x")
plt.title('Representation of Inclusive OR and its implementation using a perceptron')
plt.show()     

        
        
        
        
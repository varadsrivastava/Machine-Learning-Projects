# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:39:38 2019

@author: Varad Srivastava
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

mydata=pd.read_csv(r"LinearSeparableNEW.csv")

X_input=mydata.iloc[:,:-1]
XT_input=X_input.values.reshape(600,2)
Y_output=mydata.iloc[:,2]
YT_output=Y_output.values.reshape(600,1)

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

l_rate = 0.1
n_epoch = 5
weights = train_weights(XT_input, l_rate, n_epoch)
print(weights)

for row in XT_input:
	prediction = predict(row, weights)
	print("Predicted=%d" % (prediction))
    
   
X1 = X_input.iloc[:,0]    
X2 = X_input.iloc[:,1]   
plt.scatter(X1,X2, color='red')   
    
    







    
    
    
    



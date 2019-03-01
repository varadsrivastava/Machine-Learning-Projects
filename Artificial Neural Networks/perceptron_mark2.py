# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:14:28 2019

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
mydata_T = mydata.values.reshape(600,3)

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i] 
	return 1.0 if activation >= 0.0 else 0.0


def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = predict(row, weights)
			error = row[2] - prediction
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		print('>epoch=%d, lrate=%.3f' % (epoch, l_rate))
	return weights

l_rate = 0.01
n_epoch = 5
weights = train_weights(mydata_T, l_rate, n_epoch)
#weights[0] = 0.5
print(weights)
for row in XT_input:
	prediction = predict(row, weights)
	print("Predicted=%d" % (prediction))


#Testing a new input
test = pd.DataFrame([9.0,8.3])
test1 = test.iloc[0,0]
test2 = test.iloc[1,0]
test_T = test.values.reshape(1,2)
predict(test_T, weights)

#plotting
X1 = X_input.iloc[:,0]    
X2 = X_input.iloc[:,1]  
x_g=[8,0]
y_g=[0,8]
#for i in range(len(weights)) :
  #weights[i] = weights[i] / 10
plt.scatter(X1,X2, color='red',label ='Training')
plt.scatter(test1,test2,color='blue', label ='Our test set')
x_intercept = [-weights[0] / weights[2] ,0]
y_intercept = [0,-weights[0] / weights[1]]
plt.plot(x_g, y_g, color = 'green',label='Decision boundary of perceptron')
#plt.plot(x_intercept,y_intercept)
plt.show();





    
    







    
    
    
    



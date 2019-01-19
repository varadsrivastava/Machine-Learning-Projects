# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:39:38 2019

@author: Varad Srivastava
"""

import numpy as nm
import matplotlib.pyplot as plt
import pandas as pd

mydata=pd.read_csv(r"C:\Users\Varad Srivastava\Downloads\LinearSeparable.csv")

X_input=mydata.iloc[:,0]
Y_input=mydata.iloc[:,1]
output=mydata.iloc[:,2]

XT_input=X_input.values.reshape(599,1)
YT_input=Y_input.values.reshape(599,1)
output_T=output.values.reshape(599,1)

XT_input_train=XT_input[,420]
YT_input_train=YT_input[,420]
output_T_train=output_T[,420]

XT_input_test=XT_input[420,]
YT_input_test=YT_input[420,]
output_T_test=output_T[420,]

def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
        #activation=sum(weight_i + x_i) + bias
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

def weights():
    weights:

type(XT)




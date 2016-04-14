import numpy as np 

## Sigmoid activation function definition
def sigmoidFunct(x, deriv=False):
	if (deriv == True):
		return x*(1-x)
	return 1/(1 + np.exp(-x))

## ReLU activation function definition
def ReLU(x, deriv=False):
    if(deriv == True):
        1. *(x > 0)
    return x * (x > 0)

 ## Tanh activation function definition
def tanh(x, deriv = False):
	if(deriv == True):
		1. - x * x
	return np.tanh(x)


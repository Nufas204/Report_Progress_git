### Autoencoder and toy example
## 1 hidden layer

import numpy as np

## Sigmoid activation function definition
def sigmoidFunct(x, deriv=False):
	if (deriv == True):
		return x*(1-x)
	return 1/(1 + np.exp(-x))

# Input dataset
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
X1,X2 = X.shape

# Target or output dataset
y = X

# Seed random number to make it deterministic for future validation
np.random.seed(1)

## 2 hidden layer(1 encoder, 1 decoder)

# Initialization of alpha
alphas = [0.001, 0.01, 0.1, 0.5, 1, 10, 100]

# Iteration with alpha
for alpha in alphas:
    print ("\n Training with alpha {}".format(alpha))
    np.random.seed(1)
    
    # Randomly weights initialization
    en_syn_0 = 2*np.random.random((X2,int(X2/2))) - 1
    de_syn_0 = 2*np.random.random((int(X2/2),X2)) - 1
    
    # iteration
    krange = np.arange(60000)   
    
    # Neural network with different alpha effect
    for j in krange:
        
        ## Feed forward step ##
        en_layer_0 = X
        ac_layer = sigmoidFunct(np.dot(en_layer_0, en_syn_0))
        de_layer_0 = sigmoidFunct(np.dot(ac_layer, de_syn_0))
        
        ## Backpropagataion step ##
        de_layer_err = de_layer_0 - y
        
        # To output error calculation for every 10000 iterations 
        if (j% 10000) == 0:
            print("Error after {} iterations {}".format(j, np.mean(np.abs(de_layer_err))))
            
        de_layer_der = de_layer_err * sigmoidFunct(de_layer_0, True)
        
        ac_layer_err = de_layer_der.dot(de_syn_0.T)
        ac_layer_der = ac_layer_err * sigmoidFunct(ac_layer,True)
        
        # Updating the weights
        de_syn_0 -= alpha * (ac_layer.T.dot(de_layer_der))
        en_syn_0 -= alpha * (en_layer_0.T.dot(ac_layer_der))    
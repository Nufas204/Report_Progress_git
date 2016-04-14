### Autoencoder and toy example
## 2 Hidden layer with ReLu

import numpy as np
from activfunc import * #Import pre-defined activation function

# Input dataset
X = np.array([[0,0,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,0]])
X1,X2 = X.shape
print(X2, int(X2/2))

# Target or output dataset
y = X

# Seed random number to make it deterministic for future validation
np.random.seed(2)

## 2 hidden layer(1 encoder, 1 decoder)

# Initialization of alpha
#alphas = [0.001, 0.01, 0.1, 0.5, 1, 10, 100]
alphas = [0.9, 0.95, 2, 2.3]

# Iteration with alpha
for alpha in alphas:
    print ("\n Training with alpha {}".format(alpha))
    np.random.seed(1)
    
    # Randomly weights initialization
    en_syn_0 = 2*np.random.random((X2,int(X2/2))) - 1
    en_syn_1 = 2*np.random.random((int(X2/2),int(X2/2))) - 1
    de_syn_0 = 2*np.random.random((int(X2/2),int(X2/2))) - 1
    de_syn_1 = 2*np.random.random((int(X2/2),X2)) - 1

    # iteration
    krange = np.arange(600000)   
    
    # Neural network with different alpha effect
    for j in krange:
        
        ## Feed forward step ##
        en_layer_0 = X
        en_layer_1 = sigmoidFunct(np.dot(en_layer_0, en_syn_0))
        ac_layer   = sigmoidFunct(np.dot(en_layer_1, en_syn_1))
        de_layer_0 = sigmoidFunct(np.dot(ac_layer, de_syn_0))
        de_layer_1 = sigmoidFunct(np.dot(de_layer_0, de_syn_1))        
        
        ## Backpropagataion step ##
        de_layer_1_err = de_layer_1 - y
        
        # To output error calculation for every 10000 iterations 
        if (j% 100000) == 0:
            print("Error after {} iterations {}".format(j, np.mean(np.abs(de_layer_1_err))))
            
        de_layer_1_der = de_layer_1_err * sigmoidFunct(de_layer_1, True)
        
        de_layer_0_err = de_layer_1_der.dot(de_syn_1.T)
        de_layer_0_der = de_layer_0_err * sigmoidFunct(de_layer_0, True)

        ac_layer_err   = de_layer_0_der.dot(de_syn_0.T)
        ac_layer_der   = ac_layer_err * sigmoidFunct(ac_layer,True)
        
        en_layer_1_err = ac_layer_der.dot(en_syn_1.T)
        en_layer_1_der = en_layer_1_err * sigmoidFunct(en_layer_1, True)

        en_layer_0_err = en_layer_1_der.dot(en_syn_0.T)
        en_layer_0_der = en_layer_0_err * sigmoidFunct(en_layer_0,True)


        # Updating the weights
        de_syn_1 -= alpha * (de_layer_0.T.dot(de_layer_1_der))
        de_syn_0 -= alpha * (ac_layer.T.dot(de_layer_0_der))
        en_syn_1 -= alpha * (en_layer_1.T.dot(ac_layer_der))    
        en_syn_0 -= alpha * (en_layer_0.T.dot(en_layer_1_der))    
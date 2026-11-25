import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)


        ## converting to numpy arrays for easier calculations
        x  = np.array(x,  dtype=float)
        W1 = np.array(W1, dtype=float)
        b1 = np.array(b1, dtype=float)
        W2 = np.array(W2, dtype=float)
        b2 = np.array(b2, dtype=float)
        y_true = np.array(y_true, dtype=float)

        ## store output as dictionary
        output = {}

        ## 1. calculate forward pass
        
        ## 2 neurons in first hidden layer (2x2 W1); x dim = (2,)
        print('dim of W1, x: ', W1.shape, x.shape)
        Z1 = (x@(W1.T)) + b1 # (2x2)*(2,) + (2,)

        ## apply activation function relu
        a1 = np.maximum(0, Z1) # (2,)

        ## second hidden layer
        Z2 = (a1@(W2.T)) + b2 # (1x2)*(2,) + (1,) --> (1,) (get a single value)

        ## Z2 = y_hat
        ## 2. calculate loss; n = 1
        L = np.mean((Z2 - y_true)**2)

        output['loss'] = np.round(L, 4)

        print('DONE FORWARD PASS')
        print('shapes: ', Z1.shape, a1.shape, Z2.shape, L.shape)


        ## 3. backpropagate for gradients
        print('STARTING BACKPROP')

        ## starting at last neuron Z2
        ## dL/dZ2; L = (Z2 - y_true)^2
        n = len(y_true)
        d_Z2 = 2*(Z2 - y_true) / len(y_true) # shape (1,)
        print('Z2 shape: ', Z2.shape, 'y_true shape: ', y_true.shape)
        print('d_Z2 shape: ', d_Z2.shape)

        ## dL/db2 = dL/dZ2 * dZ2/db2
        # already have dL/dZ2 ==> global derivative
        # need local derivative dZ2/db2 = 1
        d_b2 = d_Z2 * 1 # shape (1,)

        ## dL/dW2 = dL/dZ2 * dZ2/dW2
        # local derivative dZ2/dW2 = a1; a1 shape (2,); global deriv shape (1,)
        # so put local deriv * global deriv so inner shapes line up
        # d_W2 = a1 * d_Z2 # shape (2,)
        d_W2 = np.outer(d_Z2, a1)
        print('d_W2 shape: ', d_W2.shape)


        ## dL/da1 = dL/dZ2 * dZ2/da1
        # local deriv dZ2/da1 = W1; shape (2x2)
        # global grad shape (2,)
        print('W2 shape: ', W2.shape)
        # d_a1 = W2 * d_Z2 # shape (2x2)
        d_a1 = d_Z2 @ W2
        print('d_a1 shape: ', d_a1.shape)


        ## dL/dZ1 = dL/dZ2 * dZ2/da1 * da1/dZ1
        # local deriv da1/dZ1 = deriv of relu func = 0 if f'(x) < 0, 1 if f'(x) > 0
        mask = (Z1 > 0).astype(float)
        # r_prime = Z1 * mask
        # r_prime shape (2,) (local); d_a1 = (2x2)
        d_Z1 = d_a1 * mask



        ## dL/db1 = dL/dZ2 * dZ2/da1 * da1/dZ1 * dZ1/db1
        # local deriv dZ1/db1 = 1
        d_b1 = 1 * d_Z1 # shape = (2,) * (2x2) = 

        
        ## dL/dW1 = dL/dZ2 * dZ2/da1 * da1/dZ1 * dZ1/dW1
        # local deriv dZ1/dW1 = x
        # x shape: (2,) 
        print('x: ', x, 'shape: ', x.shape)
        print('d_z1:', d_Z1, 'shape', d_Z1.shape)
        # d_W1 = x * d_Z1
        d_W1 = np.outer(d_Z1, x)

        output['dW1'] = np.round(d_W1, 4).tolist()
        output['db1'] = np.round(d_b1, 4).tolist()
        output['dW2'] = np.round(d_W2, 4).tolist()
        output['db2'] = np.round(d_b2, 4).tolist()

        return output

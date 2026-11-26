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

        ## step 1: forward pass
        z1 = (x@(W1.T)) + b1

        a1 = np.maximum(0, z1)

        z2 = (a1@(W2.T)) + b2

        L = np.mean((z2 - y_true)**2)

        print('done forward pass\n: ')
        print('z1: ', z1, '\tshape: ', z1.shape)
        print('a1: ', a1, '\tshape: ', a1.shape)
        print('z2: ', z2, '\tshape: ', z2.shape)
        print('L: ', L, '\tshape: ', L.shape)


        output['loss'] = np.round(L, 4)


        ## step 2: backpropagate; calculate weight gradients
        ## go backwards from L to z1
        ## dL/dz2 ==> used for db2, dW2, da1
        ## dL/da1 ==> used for dz1
        ## dL/dz1 ==> used for db1, dW1

        d_z2 = (2*(z2 - y_true)) / len(y_true)

        d_b2 = d_z2 

        ## dz2/dW2 = a1 ; dL/dW2 = dz2 * a1 
        ## W2 has shape (1x2) so gradient dW2 must be (1x2) --> 1 row 2 cols (2D); 
        ## z2 has shape (1,); a1 has shape = z1 = (2,) \\ both 1D, use np.outer (or reshape)
        d_W2 = np.outer(d_z2, a1) # should get shape (1, 2)

        ## dz2/da1 = W2 ; dL/da1 = dz2 * W2
        ## a1 has shape (2,); da1 must have (2,)
        ## W2 shape = (1x2), z2 shape = (1, ) \\ 2D, 1D; np converts 1D into row
        ## temp d_z2 shape = (1,1) @ (1,2) --> (1,2) --> drop added dim ==> (2,)
        d_a1 = d_z2 @ W2


        ## da1/dz1 = deriv of relu func = 0 if z1 < 0, 1 if z1 > 0
        ## dL/dz1 = dl/da1 * da1/dz1
        ##  ReLU is applied independently to each neuron (1-to-1), so its derivative must be multiplied element-by-element (*), not as an outer product (np.outer, which turns your 1D (2,) vector into a 2D (2, 2) matrix!).
        d_relu = (z1 > 0).astype(float) # a true/false arr 
        d_z1 =  d_a1 * d_relu


        ## dz1/db1
        d_b1 = d_z1

        
        ## dz1/dW1 = x
        ## x has shape (2,), z1 has shape (2,) \\ 1D, 1D, use outer
        ## z1 needs to be rows (neurons), x needs to be cols (inputs)
        d_W1 = np.outer(d_z1, x)
        

        output['dW1'] = np.round(d_W1, 4).tolist()
        output['db1'] = np.round(d_b1, 4).tolist()
        output['dW2'] = np.round(d_W2, 4).tolist()
        output['db2'] = np.round(d_b2, 4).tolist()

        return output

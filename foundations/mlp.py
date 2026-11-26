import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        

        ## given ndarry; don't need to convert yet
        ## x.shape: (2,)
        ## weights[0] = layer 1's weights = (2x2)
        ## weights[1] = layer 2's weights = (2x1)
        ## biases[0] = layer 1's biases = (2,)
        ## biases[1] = layer 2's bias = (1,)
        ## output.shape = (1,)

        ## 2 inputs --> 2 nodes --> 2 nodes --> output

        ## FORWARD PASS
        print('starting forward pass')

        ## formula: h[i] = ReLU(h[i-1] @ W[i] + b[i])
        ## looping over array; starting with h = x

        h = x

        for i in range(len(biases)):
            W_i = weights[i]
            b_i = biases[i]

            z = h @ W_i + b_i


            if i != len(biases)-1:
                h = np.maximum(0, z)
            else:
                h = z

        return np.round(h, 5)  





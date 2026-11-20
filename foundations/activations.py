import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)

        ## doesn't return ndrray type; returns a list instead
        # result = []
        # for i in z:
        #     k = -1*i
        #     sig = 1 / (1 + np.exp(k))

        #     result.append(np.round(sig, 5))
        
        # return result

        ## just return the func applied to the ndarray since it applies element-wise operation
        return np.round(1 / (1 + np.exp(-z)), 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise

        ## same tihng as above
        # result = []

        # for i in z:
        #     j = np.maximum(0, i)
        #     result.append(np.round(j, 5))

        # return result

        return np.round(np.maximum(0, z), 5)

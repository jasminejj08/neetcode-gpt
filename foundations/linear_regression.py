import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places

        ## forward pass (y_hat = X . W); dot product of feature matrix and weight vector

        dp = np.dot(X, weights)


        ## return ndarray
        return np.round(dp, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places

        n = len(ground_truth)

        sum = 0
        for i in range(n):
            sum += (model_prediction[i] - ground_truth[i])**2
        
        mse = (sum/n)[0]

        
        ## return float
        return np.round(mse, 5)
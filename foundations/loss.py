import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)

        clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        n = len(y_true)
        
        sum = 0
        for i in range(n):
            sum += y_true[i] * np.log(clipped[i]) + (1-y_true[i]) * np.log(1 - clipped[i])
        
        result = -1 * 1/n * sum

        ## return a float (singular value)

        return np.round(result, 4)


    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)

        clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        n = len(y_true)

        sum = 0

        for i in range(n):
            arr = y_true[i]
            for j in range(len(arr)):
                sum += y_true[i][j] * np.log(clipped[i][j])
        
        result = -1 * 1/n * sum



        return np.round(result, 4)

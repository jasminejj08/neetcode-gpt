class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        # if num iterations = 0; return initial value unchanged
        if iterations == 0: 
            return round(init, 5)
        # otherwise repeat update rule for num of iterations
        else:
            for i in range(iterations):

                # x_new = x_old - (learning_rate * prime)
                x_new = init - (learning_rate * (2*init))
                init = x_new

            return round(x_new, 5)

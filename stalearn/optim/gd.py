import numpy as np
from .optimizer import Optimizer

class GD(Optimizer):
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        super().__init__(learning_rate=learning_rate, max_iters=max_iters, tol=tol)
        self.params = None
    
    def step(self, gradient):
        return -self.learning_rate * gradient
    
    def optimize(self, gradient_func, init_params):
        params = init_params
        for _ in range(self.max_iters):
            gradient = gradient_func(params)
            update = self.step(gradient)

            if np.linalg.norm(update) < self.tol:
                break

            params += update
        
        return params

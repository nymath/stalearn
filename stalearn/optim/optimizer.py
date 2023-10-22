from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate=0.01, max_iters=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol

    @abstractmethod
    def step(self, gradient):
        pass

    @abstractmethod
    def optimize(self, gradient_func, init_params):
        pass
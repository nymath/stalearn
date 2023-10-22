from typing import Callable
import numpy as np
from scipy.optimize import minimize
import torch 
import torch.nn as nn

from ..utils.functools import create_function_from_string

class BaseMLE:
 
    def __init__(self, data: np.ndarray, pdf: Callable):
        self.data = data 
        self.pdf = pdf
        self.vectorized_pdf = np.vectorize(pdf)
        self.n_params = pdf.__code__.co_argcount - 1
        self.fitted = False

    def negative_log_likelihood(self, params):
        return -np.sum(np.log(self.pdf(self.data, *params)))

    def fisher_information_matrix(self):
        # Ensure that the Fisher Information Matrix is computed only after fitting
        if not self.fitted:
            raise ValueError("You should fit the model first before computing the Fisher Information Matrix.")
        
        def _replace_np_with_torch(func):
            import inspect
            import textwrap

            source_code = inspect.getsource(func)
            modified_code = source_code.replace('np.', 'torch.')
            modified_code = '\n'.join([line for line in modified_code.split('\n') if not line.strip().startswith('@')])
            modified_code = textwrap.dedent(modified_code)

            local_namespace = {}
            exec(modified_code, globals(), local_namespace)
            func_obj = local_namespace[func.__name__]
            return func_obj 
        
        torch_pdf = _replace_np_with_torch(self.pdf)
        score_functions = []
        torch_data = torch.tensor(self.data)
        torch_params = torch.tensor(self.params, requires_grad=True)
        for data in torch_data:
            pdf_output = torch_pdf(data, *torch_params)
            loss = torch.log(pdf_output)
            grad = torch.autograd.grad(loss, torch_params, retain_graph=True, create_graph=True)
            score_functions.append(grad[0].reshape(-1, 1))
        
        score_functions = torch.concat(score_functions, axis=1)

        return torch.matmul(score_functions, score_functions.T).detach().numpy()

    
    def fit(self, initial_guess, bounds=None, method='L-BFGS-B', tol=1e-9, calculate_variance=False):
        assert len(initial_guess) == self.n_params
        if bounds is not None:
            assert len(bounds) == self.n_params
        result = minimize(self.negative_log_likelihood, initial_guess, bounds=bounds, method=method, tol=tol)
        if result.success:
            self.params = result.x
            self.fitted = True
            self.params_var = None
            if calculate_variance:
                self.params_var = np.diag(np.linalg.inv(self.fisher_information_matrix()))
            return self.params
        else:
            raise ValueError(result.message)
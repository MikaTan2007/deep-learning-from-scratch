import numpy as np
from numpy import ndarray
from typing import Callable

def square (x: ndarray) -> ndarray:
    #returns squared function
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)

def deriv(func: Callable[[ndarray], ndarray],
          input_: ndarray,
          delta: float = 0.001) -> ndarray:
    
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)


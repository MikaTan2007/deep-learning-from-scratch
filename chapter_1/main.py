import numpy as np
from numpy import ndarray

def square (x: ndarray) -> ndarray:
    return np.power(x, 2)

def leaky_relu(x: ndarray) -> ndarray:
    return np.maximum(0.2 * x, x)
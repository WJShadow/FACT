import numpy as np  
from numba import njit, prange
  
@njit(parallel=True, cache=True, fastmath=True, nogil=True)
def compute_std(array: np.ndarray) -> np.ndarray:  
    T, H, W = array.shape  
    result = np.zeros((H, W), dtype=array.dtype)  
      
    for i in prange(H):  
        for j in prange(W):  
            mean = np.mean(array[:, i, j])  
            variance = np.mean((array[:, i, j] - mean) ** 2)  
            result[i, j] = np.sqrt(variance)  
    result = (result - result.min()) / (result.max() - result.min()) * 255
    return result  
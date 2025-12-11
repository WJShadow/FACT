import numpy as np  
from numba import njit, prange  
  
@njit(parallel=True, cache=True)  
def thresh_max(raw_vid, threshold):  
    """  
    Set all values greater than the threshold to threshold
  
    Inputs:
    -------  
        **raw_vid** TxHxW ndarray: raw video in ndarray
        **threshold ** float: threshold 
  
    Outputs:  
    --------
        **raw_vid** TxHxW ndarray: processed raw video
    """  
    T, H, W = raw_vid.shape  
    for t in prange(T): 
        for h in range(H):  
            for w in range(W):  
                if raw_vid[t, h, w] > threshold:  
                    raw_vid[t, h, w] = threshold  
    return raw_vid

@njit(parallel=True, cache=True)  
def thresh_min(raw_vid, threshold):  
    """  
    Set all values greater than the threshold to threshold
  
    Inputs:
    -------  
        **raw_vid** TxHxW ndarray: raw video in ndarray
        **threshold ** float: threshold 
  
    Outputs:  
    --------
        **raw_vid** TxHxW ndarray: processed raw video
    """  
    T, H, W = raw_vid.shape  
    for t in prange(T): 
        for h in range(H):  
            for w in range(W):  
                if raw_vid[t, h, w] < threshold:  
                    raw_vid[t, h, w] = threshold  
    return raw_vid
  
@njit(parallel=True, cache=True)  
def thresh_binary(input_array, threshold):  
    """  
    Set all values greater than the threshold to 1, and all values smaller than threshold to 0
  
    Inputs:
    -------   
        **raw_vid** TxHxW ndarray: raw video in ndarray
        **threshold ** float: threshold 
  
    Outputs:  
    -------- 
        **raw_vid** TxHxW ndarray: processed raw video
    """  
    T, H, W = input_array.shape  
    for t in prange(T):  
        for h in range(H):  
            for w in range(W):  
                if input_array[t, h, w] > threshold:  
                    input_array[t, h, w] = 1  
                else:  
                    input_array[t, h, w] = 0  
    return input_array  
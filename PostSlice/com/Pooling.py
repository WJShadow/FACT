import numpy as np  
from numba import njit, prange  
  
@njit(parallel=True, cache=True)  
def average_pooling(input_array, window_size):  
    '''
    Accelerated average pooling along time axis
    
    Inputs:
    -------
        input_array ndarray HxWxT: The input array to be down-sampled
        window_size int: window size for performing average-pooling 
        
    Outputs:
    --------
        output_array ndarray HxWxoutput_T: average pooled 

    
    '''
    H, W, T = input_array.shape  
      
    output_T = (T - window_size) // window_size + 1    
    output_array = np.zeros((H, W, output_T))  
      
    for h in prange(H):  
        for w in prange(W):  
            for t in prange(output_T):  
                start = t * window_size  
                end = start + window_size  
                output_array[h, w, t] = np.mean(input_array[h, w, start:end])  
      
    return output_array  
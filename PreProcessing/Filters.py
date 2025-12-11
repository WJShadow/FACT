import numpy as np
from scipy.ndimage import median_filter
from joblib import Parallel, delayed
  
def apply_median_filter_to_slice(input_slice, core_size):  
    '''
    Function for applying median filter to a single slice of picture, mainly for parallel processing 
    
    '''
    filtered_slice = median_filter(input_slice, size=core_size)
    return filtered_slice
  
def median_filter_parallel(raw_vid, core_size):
    """  
    Function for applying median_filter in parallel to the whole video
  
    Inputs:
    -------  
        **raw_vid**: TxHxW ndarray: raw video in ndarray
        **a** int: size of the filter core 
  
    Outputs:
    ------- 
        raw_vid: The same variable as the input array, with each image filtered using the median-filter
    """  
    
    if len(raw_vid.shape) != 3:  
        raise ValueError("Input sequence should have three dimensions [T,H,W]")  
    T, H, W = raw_vid.shape  
  
    # Process each slice of image in parallel 
    filtered_slices = Parallel(n_jobs=-1, verbose=10)(  
        delayed(apply_median_filter_to_slice)(raw_vid[t], core_size) for t in range(T)  
    )  
    filtered_vid = np.stack(filtered_slices, axis=0)  
    # Stack all processed images to the output array 
    # output_array = np.stack(output_slices, axis=0)  
  
    return filtered_vid
  
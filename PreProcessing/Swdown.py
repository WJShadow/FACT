import numpy as np  
from scipy.ndimage import convolve1d 
import time 
  
def process_video(vid, W, DW):  
    ''' SW avg and tunable DW
    
    '''
    if not isinstance(vid, np.ndarray) or vid.ndim != 3:  
        raise ValueError("float16 3D ndarray required")  
      
    T, H, _W = vid.shape  
    if W <= 0 or W > T:  
        raise ValueError(" SW size >0 and <T ")  
    if DW < 0:  
        raise ValueError(" DW > 0 ")  
    
    window = np.ones(W) / W  

    averaged_vid = np.zeros((T, H, vid.shape[2]), dtype=np.float16)  
    for h in range(H):  
        for w in range(vid.shape[2]):  
            averaged_vid[:, h, w] = convolve1d(vid[:, h, w], window)  

    if DW == 0:  
        downsampled_vid = averaged_vid  
    else:  
        downsampled_vid = averaged_vid[::DW, :, :]  
      
    return downsampled_vid  

import numpy as np
from scipy.ndimage import convolve1d
import time

def std_video(vid, W, DW):

    if not isinstance(vid, np.ndarray) or vid.ndim != 3:
        raise ValueError('float16 3D ndarray required')
    T, H, _W = vid.shape
    if W <= 0 or W > T:
        raise ValueError(' SW size >0 and <T ')
    if DW < 0:
        raise ValueError(' DW > 0 ')
    

    window = np.ones(W) / W
    

    averaged_vid = np.zeros((T, H, vid.shape[2]), dtype=np.float16)
    for h in range(H):
        for w in range(vid.shape[2]):
            data = vid[:, h, w].astype(np.float32)  
            mean_x = convolve1d(data, window, mode='reflect')
            mean_x2 = convolve1d(data**2, window, mode='reflect')
            std = np.sqrt(mean_x2 - mean_x**2).astype(np.float16)  
            averaged_vid[:, h, w] = std
    

    if DW == 0:
        downsampled_vid = averaged_vid
    else:
        downsampled_vid = averaged_vid[::DW, :, :]
    
    return downsampled_vid


  
def downmax_video(vid, DW):  
    ''' 
    SW downsamp of max
    '''

    if not isinstance(vid, np.ndarray) or vid.ndim != 3:  
        raise ValueError("float16 3D ndarray required")  
      
    H, W, T = vid.shape  
    if W <= 0 or W > T:  
        raise ValueError(" SW size >0 and <T ")  
    if DW < 0:  
        raise ValueError(" DW > 0 ")  
  

    window = np.ones(W) / W  
      
    downdim = np.int16(np.ceil(T/DW))

    downmax_vid = np.zeros((H, W, downdim), dtype=np.uint8)  
    for t in range(downdim):  
        downmax_vid[:,:,t] = np.max(vid[:,:,t*downdim:(t+1)*downdim], axis=2)
    
    return downmax_vid



if __name__ == "__main__":  
    # Create a sample video with shape (T, H, W)  
    np.random.seed(0)  
    vid = np.random.rand(1007, 512, 512)  # Example video with 100 frames, 240x320 resolution  
      
    # Define window length and downsampling interval  
    W = 10
    DW = 10  
    
    st = time.time()
    # Process the video  
    result = process_video(vid, W, DW)  
    print(f'Total time: {time.time()-st} s')
      
    print(f"Processed video shape: {result.shape}")
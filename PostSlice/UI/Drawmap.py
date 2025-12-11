import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.sparse as sp
import numba
from numba import prange
from numba import jit
import time

def get_random_color(num_masks):
    '''Randomly generate a list of three int ranging from 0 to 255, indicating the R,G,B intensity correspondingly

    Inputs: None
    -------
    
    Outputs:
    -------
        **rand_color** 3 list [int32]: the list of RGB color intensity
    '''
    rand_color = list(np.random.choice(range(256), size=(num_masks,3)))
    rand_color = np.uint8(rand_color)
    # print(color)
    return rand_color

def draw_mask_colormap(masks, image_shape):  # code trying to use direct-reshape as the method for adding color to the final mask for reducing time
    '''Drawing segmented masks using randomly selected color

    Inputs: 
    -------
        **masks** dok_matrix NxP[int32]: sparse dok_matrix storing the N masks in vector form, P is the number of pixels
        **image_shape** tuple (int32, int32): shape of original image in pixels, which is (NxM)
    
    Outputs:
    -------
        **colormap** array NxMx3 [uint8]
    '''

    # color_img = np.zeros((image_shape[0]*image_shape[1],3), dtype=np.uint8)
    N_masks = masks.shape[0]
    rand_color = get_random_color(N_masks)  # generate a random color in the form of [R,G,B] for each independent mask and store them in the same place for further processing 
    (mask_labs, mask_inds) = masks.nonzero()

    color_arr = fast_drawcmap(mask_labs, mask_inds, rand_color, masks.shape[1])

    color_img = color_arr.reshape((image_shape[0],image_shape[1],3))  # reshape the output image to the correct image shape for the sake of further plt.imshow to show as the color map

    return color_img

# ---------------------------------- Fast drawer for generating colormap ----------------------------
@jit("u1[:,:](i4[:],i4[:],u1[:,:],i8)", nopython=True, fastmath=True, cache=True, parallel=True)
# @jit( nopython=True, fastmath=True, cache=True, parallel=True)
def fast_drawcmap(mask_labs_arr, mask_inds_arr, colors, pixel_len):
    '''
    Func for rapidly paint masks with different random colors

    Inputs:
    -------
        **mask_labs_arr**: array of mask labels
        **mask_inds_arr**: array of mask indices
        **colors**: Nx3 array storing random 3-channel RGB values for each mask
        **pixel_len**: number of pixels in a 2-D image 

    '''
    color_arr = np.zeros((pixel_len,3), dtype=np.uint8)
    for pointer in range(mask_inds_arr.shape[0]):
        curr_lab = mask_labs_arr[pointer]
        curr_ind = mask_inds_arr[pointer]
        color_arr[curr_ind,0] += colors[curr_lab,0]
        color_arr[curr_ind,1] += colors[curr_lab,1]
        color_arr[curr_ind,2] += colors[curr_lab,2]
    
    return color_arr



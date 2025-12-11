import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from math import sqrt
import numba
from numba import prange
from numba import jit
import time


def efi_calc_COM(masks, H, W):
    '''The accelerated version of 'calc_COM', calling numba accelerated dict_COM to efficiently obtain the COM of a bunch of masks
       and having the same inputs and outputs as 'calc_COM' 

    Inputs:
    --------
        **masks** dok_matrix NxP [int32]: sparse dok_matrix storing the N masks in vector form, P is the number of pixels
        **H** [int64]: Height of the original image
        **W** [int64]: Width of the original image

    Outputs:
    --------
        **COMs** nparray Nx2 [int32]: numpy array storing the COM coordinate of each mask

    '''
    N_mask = masks.shape[0]

    posis = []
    values = []
    for posi,value in masks.items():
        posis.append(posi)
        values.append(value)
    posis_arr = np.asarray(posis)

    COMs = dict_COM(posis_arr, N_mask, H, W)
    
    return COMs



# code for generating the COMs efficiently
@jit("i4[:,:](i4[:,:], i8, i8, i8)", nopython=True, fastmath=True, parallel=True, cache=True)
def dict_COM(dictarry_masks, N_mask, H, W):
    '''dictionary-format accelerated COM calculation, the Output of which is already in the format of a Nx2 numpy array 
    
    Inputs:
    --------
        **dictarry_masks** nparray Mx2 [int32]: array format dictionary storing positions of each mask pixel
        **N_mask** [int64]: number of masks stored in the dictarry_masks
        **H** [int64]: Height of the original image
        **W** [int64]: Width of the original image

    Outputs:
    --------
        
        **COMs** nparray Nx2 [int32]: Numpy array of the COM coordinates in the format of (H_COM, W_COM)
        Note: when using this dictionary-accelerated COM caculation code for fast COM, the code in 'evaluation_Euclidean' should be adjusted in compatibility with the fast method's input and output
        
    '''

    COMs = np.zeros((N_mask,2), dtype=np.int32)  # array for storing the COM positions
    
    sum_h_array = np.zeros(N_mask, dtype=np.int32)  # array for summing the sub value of pixels in each mask
    sum_w_array = np.zeros(N_mask, dtype=np.int32)
    num_array = np.zeros(N_mask, dtype=np.int32)
    
    # Summing the ind of pixels in each mask
    for iter_pix in prange(dictarry_masks.shape[0]):
        h_pix = dictarry_masks[iter_pix,1] // W
        w_pix = dictarry_masks[iter_pix,1] - h_pix*W
        sum_h_array[dictarry_masks[iter_pix,0]] += h_pix
        sum_w_array[dictarry_masks[iter_pix,0]] += w_pix

        num_array[dictarry_masks[iter_pix,0]] += 1

    for iter_mask in prange(N_mask):
        if num_array[iter_mask] == 0:
            sub_h = -1
            sub_w = -1
        else:
            # COM_ind = np.int32(round(np.float32(sum_array[iter_mask]) / np.float32(num_array[iter_mask])))  # calculating the linear ind position of COM
            # sub_h = COM_ind // W  # converting the ind form of COM position to sub form:  (sub_h, sub_w)
            # sub_w = COM_ind - sub_h*W
            sub_h = np.int32(round(np.float32(sum_h_array[iter_mask]) / np.float32(num_array[iter_mask])))
            sub_w = np.int32(round(np.float32(sum_w_array[iter_mask]) / np.float32(num_array[iter_mask])))
        COMs[iter_mask,0] = sub_h
        COMs[iter_mask,1] = sub_w
    
    return COMs
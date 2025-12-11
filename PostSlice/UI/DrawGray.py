import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.sparse as sp
import numba
from numba import prange
from numba import jit
import time

def draw_mask_grayscale(masks, image_shape):
    '''Drawing segmented masks using the value of that pixel stored in the dok matrix as the factor of its grayscale

    Inputs: 
    -------
        **masks** dok_matrix NxP[int32]: sparse dok_matrix storing the N masks in vector form, P is the number of pixels
        **image_shape** tuple (int32, int32): shape of original image in pixels, which is (NxM)
        
    Outputs:
    -------
        **colormap** array NxMx3 [uint8]
    '''

    (H,W) = image_shape
    color_img = np.zeros((H,W,3), dtype=np.uint8)
    N_masks = masks.shape[0]

    for iter_mask in range(N_masks):
        onemask_posilist = list(masks[iter_mask].keys())
        

        # rand_color = get_random_color()  # generate a random color in the form of [R,G,B] for each independent mask

        onemask_dokarr = masks[iter_mask]
        print(onemask_dokarr.shape)
        onemask_mat = onemask_dokarr.reshape((H,W)).toarray()
        print(onemask_mat.shape)
        onemask_imgmat = np.uint8(onemask_mat * 255)
        print(onemask_imgmat.shape)

        onemask_rgbmat = np.repeat(np.expand_dims(onemask_imgmat,2), 3, axis=2)
        print(onemask_rgbmat.shape)
        color_img += onemask_rgbmat

    
    return color_img

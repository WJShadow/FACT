import hdf5storage as scio
import scipy.sparse as sp
import numpy as np

def read_mat_mask(mask_pth, mask_shape_2d):
    masks_mat=scio.loadmat(mask_pth)

    masks = sp.vstack((sp.lil_array(tomask(np.hstack((msk[0][0][0].T, msk[0][0][1].T)), mask_shape_2d)) for msk in masks_mat['masks'][0]), format='lil')

    return masks


 # Transform masks from coords to 
def tomask(coords, maskshape_2d):  # modified function for generating masks from the stored pixel coordinates 
    mask = np.zeros(maskshape_2d)  # 
    coords_arry = np.asarray(coords)  
    posis = tuple([coords_arry[:,0], coords_arry[:,1]])  
    mask[posis] = 1  
    mask = mask.reshape((1,-1))  
    return mask
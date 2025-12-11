import numpy as np
import scipy.sparse as sp
from numba import njit, prange

def COM_sp(masks: sp.csr_matrix, imgshape_2d: np.ndarray):
    '''Rapidly compute the COM  of the given masks in sparse matrix
    Inputs:
    -------
        **masks** sparse matrix nxP: masks in sparse matrix format
        **imgshape_2d** tuple(int,int): shape of 2-dimensional image

    Outputs:
    --------
        **COMs** ndarray nx2: center of mass of 
        
    '''
    num_masks = masks.shape[0]
    (mask_ind, posi_ind) = masks.nonzero()
    COMs = np.zeros((num_masks,2), dtype=np.float32)
    com_paral_calc(mask_ind=mask_ind, posi_ind=posi_ind, COMs=COMs, \
                    col_num=imgshape_2d[1], mask_num=num_masks)
    
    return COMs

@njit("void(i4[:],i4[:],f4[:,:],i4,i4)",parallel=True, cache=True, fastmath=True, nogil=True)  
def com_paral_calc(mask_ind, posi_ind, COMs, col_num, mask_num):  
    '''Parallely calculate the COM of each mask
    
    Inputs:
    -------
        **mask_ind** ndarray n: indices of mask numbers
        **posi_ind** ndarray n: indices of position ind (squeezed coordinates)
        **COMs** ndarray nx2: ndarray to store the calculated COMs of masks
        **col_num** int: number of columns in the iamge
        **mask_num** int: number of masks segmented form the image
    
    '''

    num_pix_arr = np.zeros(mask_num, dtype=np.int32)
    
    for i_pix in prange(mask_ind.size):  # Iteratively add each pixel
        COMs[mask_ind[i_pix],0] += (posi_ind[i_pix] // (col_num))
        COMs[mask_ind[i_pix],1] += (posi_ind[i_pix] % (col_num))
        num_pix_arr[mask_ind[i_pix]] += 1
    
    for i_mask in prange(mask_num):
        COMs[i_mask,0] /= num_pix_arr[i_mask]
        COMs[i_mask,1] /= num_pix_arr[i_mask]
    
    return 



def vhradio_sp(masks: sp.csr_matrix, imgshape_2d: np.ndarray):
    '''Rapidly compute the COM  of the given masks in sparse matrix
    Inputs:
    -------
        **masks** sparse matrix nxP: masks in sparse matrix format
        **imgshape_2d** tuple(int,int): shape of 2-dimensional image

    Outputs:
    --------
        **vhrads** ndarray nx2: the maximum x and y raidos of all masks 
        **maxmin** ndarray nx4: [min_h, min_w, max_h, max_w] of all masks
        
    '''
    num_masks = masks.shape[0]
    (mask_ind, posi_ind) = masks.nonzero()
    vhrads = np.zeros((num_masks,2), dtype=np.float32)
    vhmaxmin = np.zeros((num_masks,4), dtype=np.float32)
    vhmaxmin[:,0:2] = 100000
    vhrad_paral_calc(mask_ind=mask_ind, posi_ind=posi_ind, vhrads=vhrads, vhmaxmin=vhmaxmin, \
                    col_num=imgshape_2d[1], mask_num=num_masks)
    
    return vhrads, vhmaxmin

@njit("void(i4[:],i4[:],f4[:,:],f4[:,:],i4,i4)",parallel=True, cache=True, fastmath=True, nogil=False)  
def vhrad_paral_calc(mask_ind, posi_ind, vhrads, vhmaxmin, col_num, mask_num):  
    '''Parallely calculate the v and h maxradios of each mask
    
    Inputs:
    -------
        **mask_ind** ndarray n: indices of mask numbers
        **posi_ind** ndarray n: indices of position ind (squeezed coordinates)
        **COMs** ndarray nx2: ndarray to store the calculated COMs of masks
        **col_num** int: number of columns in the iamge
        **mask_num** int: number of masks segmented form the image
    
    '''

    # num_pix_arr = np.zeros(mask_num, dtype=np.int32)
    
    for i_pix in range(mask_ind.size):  # Iteratively add each pixel
        h = (posi_ind[i_pix] // (col_num))
        w = (posi_ind[i_pix] % (col_num))
        vhmaxmin[mask_ind[i_pix],0] = vhmaxmin[mask_ind[i_pix],0] if vhmaxmin[mask_ind[i_pix],0]<h else h
        vhmaxmin[mask_ind[i_pix],2] = vhmaxmin[mask_ind[i_pix],2] if vhmaxmin[mask_ind[i_pix],2]>h else h
        vhmaxmin[mask_ind[i_pix],1] = vhmaxmin[mask_ind[i_pix],1] if vhmaxmin[mask_ind[i_pix],1]<w else w
        vhmaxmin[mask_ind[i_pix],3] = vhmaxmin[mask_ind[i_pix],3] if vhmaxmin[mask_ind[i_pix],3]>w else w
    
    for i_mask in prange(mask_num):
        vhrads[i_mask,0] = vhmaxmin[i_mask,2] - vhmaxmin[i_mask,0]
        vhrads[i_mask,1] = vhmaxmin[i_mask,3] - vhmaxmin[i_mask,1]
    
    return 



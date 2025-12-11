import numpy as np
import scipy.sparse as sp
from numba import njit, prange

def merge_COM(masks: sp.csr_matrix, COMs:np.ndarray, timestamps:np.ndarray, max_dist=0.0, valid_arr=None): 
    '''Merge masks with close COM distances determined by max_dist

    Inputs:
    -------
        **masks** sparse matrix nxP: masks segmented from slice_connect
        **COMs** ndarray n: centers of corresponding segmented masks
        **timestamps** ndarray n: timestamps of corresponding segmented masks
        **max_dist** : maximum distance of COMs for two masks to be paired and merged
        **valid_arr** ndarray n: valid labels of masks which are ready to be merged
    
    Outputs:
    --------
        **masks_merged** sparse matrix mxP: merged masks 
        **timestamps_merged** list[ndarray]: list of array that stores the timestamps of each merged mask
    '''
    max_dist = np.float32(max_dist)
    num_masks_init = masks.shape[0] 

    if valid_arr is None:
        valid_arr = np.ones(num_masks_init, dtype=np.uint8)
    
    valid_list_init = valid_arr.nonzero()[0]

    if num_masks_init>1:
        iter_base = 0

        print('Finding mask pairs')
        for cent_mask in valid_list_init:
            if not valid_arr[cent_mask]:
                continue
            pair_arr = np.zeros(num_masks_init, dtype=np.uint8)
            # print('Finding pair dist')
            find_pair_dist(coords=COMs, valid_arr=valid_arr, coord=COMs[cent_mask], pair_arr=pair_arr, thresh_dist=max_dist)
            # print('Done Finding pair dist')
            pair_list = pair_arr.nonzero()[0]
            valid_arr[pair_list] = 0

            sp_row_ind = iter_base*np.ones(pair_list.size)
            sp_col_ind = pair_list

            if iter_base == 0:
                row_ind_all = sp_row_ind
                col_ind_all = sp_col_ind
            else:
                row_ind_all = np.hstack([row_ind_all, sp_row_ind])
                col_ind_all = np.hstack([col_ind_all, sp_col_ind])
            iter_base += 1

        print('Calculating transfered masks')
        transfer_mat = sp.csr_matrix(([1]*row_ind_all.size, (row_ind_all, col_ind_all)), (iter_base, num_masks_init))
        masks_merged = transfer_mat * masks # Add merged neurons using sparse matrix multiplication
        timestamps_merged = [timestamps[transfer_mat[x].nonzero()[1]] for x in range(iter_base)]

    else:
        masks_merged = masks
        timestamps_merged = timestamps
    
    return masks_merged, timestamps_merged


@njit("void(f4[:,:],u1[:],f4[:],u1[:],f4)",parallel=True, cache=True, fastmath=True, nogil=True)  
def find_pair_dist(coords, valid_arr, coord, pair_arr, thresh_dist):  
    '''Rapidly identify masks with COM distancse to target smaller than thresh_dist
    
    Inputs:
    -------
        **coords* ndarray nx2: coordinates of all masks
        **valid_list** ndarray n: status of all masks
        **coord** ndarray 2: coordinate of current central mask
        **pair_list** ndarray n: pair status of all masks to current central mask
        **thresh_dist** : threshold of distance to pair masks to current central mask 
    
    '''
    thresh_dist_square = thresh_dist**2  
    n = coords.shape[0]  
    for i in prange(n):  
        if valid_arr[i]:
            dist_square = (coords[i, 0] - coord[0])**2 + (coords[i, 1] - coord[1])**2
            if dist_square <= thresh_dist_square:  
                pair_arr[i] = 1  
  


import numpy as np
import scipy.sparse as sp
import numba
from numba import jit
import time


def calc_olratio(masks: sp.dok_matrix):
    '''Calculation of the overlap ratio between masks
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form

    Outputs:
    --------
        **olratio_map** ndarray NxN [float]: overlap-ratio matrix, with [i,j] member storing the overlap(i,j) / area(j)
    
    '''
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    area_mat = np.expand_dims(area_arr, axis=0).repeat(masks.shape[0], axis=0)  # matrix form of mask areas for division
    olratio_mat = overlap_mat / area_mat
    olratio_mat = olratio_mat - np.diag(olratio_mat.diagonal())  # matrix storing the overlapping ratio between each pair of masks
    #--Note: the [i,j] member of the matrix stands for the overlap(i,j) / area(j)

 
    return olratio_mat


@jit("i4[:](f8[:,:],f8)", nopython=True, fastmath=True, cache=True)
def find_mergepairs(olratio_mat, olratio_thresh):
    '''
    Find pairs of masks to merge according to the olratio matrix and olratio threshold
    Inputs:
    -------
        **olratio_mat** ndarray NxN [float64]: matrix stroing the overlap ratio of each mask pair
        **olratio_thresh** float64: threshold for identifing all paired masks in the olratio_mat
    
    Outputs:
    --------
        **maskpair_labs** ndarray N [int32]: label of mask group each mask belongs to
        
    '''
    num_mask = olratio_mat.shape[0]
    olstatus_mat = olratio_mat >= olratio_thresh  # Generate the status mat using the threshold

    olstatus_direcsum = np.vstack((olstatus_mat.sum(axis=0), olstatus_mat.sum(axis=1)))  # Calculate the summation of status along different axis
    olstatus_allsum = olstatus_direcsum.sum(axis=0)

    maskpair_labs = np.zeros(shape=num_mask, dtype=np.int32)  # Create the array storing the group labels to output
    maskpair_labs[:] = -1  # label all unprocessed masks using -1 
    iter_label = 0

    for iter_mask in range(num_mask):  # Iteratively process each mask 
        if (maskpair_labs[iter_mask]!=-1):  # meaning the mask has been clustered
            continue
        if (olstatus_allsum[iter_mask]==0): # meaning no pair has been found for this mask
            maskpair_labs[iter_mask] = iter_label
            iter_label += 1
            continue
        
        # Arriving here means the mask has not been processed and has at least one paired partner
        flag_exit = False
        list_group = []

        list_current, list_add = [np.int64(x) for x in range(0)], [np.int64(x) for x in range(0)]
        list_current = np.where(np.logical_or(olstatus_mat[iter_mask,:], olstatus_mat[:,iter_mask]))[0]
        list_current = list(list_current)
        while(not flag_exit): # Iteratively search for the paired masks until all matches are recorded
            list_group.extend(list_current)
            for iter_current in list_current: # Iteratively check masks in list_current
                pairedmasks = np.where(np.logical_or(olstatus_mat[iter_current,:], olstatus_mat[:,iter_current]))[0]
                if (not len(pairedmasks) > 0):
                    continue
                for iter_pair in pairedmasks:
                    if (not ((iter_pair in list_group) or (iter_pair in list_current) or (iter_pair in list_add))):  # meaning the mask has not been identified
                        list_add.append(iter_pair)  # add this mask to the addlist
                    olstatus_mat[iter_pair, iter_current], olstatus_mat[iter_current, iter_pair] = False, False  # clear the footprint in the olstatus matrix
            
            if (not len(list_add) > 0):
                flag_exit = True
            list_current = list_add.copy()
            list_add = [np.int64(x) for x in range(0)]
        
        for iter_group in list_group:  # label every mask in the group using the same label
            maskpair_labs[iter_group] = iter_label
        iter_label += 1

    return maskpair_labs
        
                


def merge_masks(masks_init: sp.dok_matrix, grouparr):
    '''
    Merge masks according to their labels in the grouparr array
    
    Inputs:
    -------
        **masks_init** dok_matrix NxP [int]: all masks in dok_matrix form
        **grouparr** ndarray N [int64]: label of each mask indicating the group to which it should be merged
    
    Outputs:
    --------
        **masks_merged** dok_matrix MxP [int]: merged masks in dok_matrix format
    '''
    masks_init = masks_init.tocsr()
    masks_num = np.max(grouparr)+1
    masks_merged = sp.dok_matrix((0, masks_init.shape[1]), dtype=masks_init.dtype)
    for iter_merge in range(masks_num):
        # time_merge_st = time.time()
        # masks_merged[iter_merge,:] = sp.dok_matrix(masks_init[np.where(grouparr==iter_merge)].sum(axis=0))
        mask_arr = masks_init[np.where(grouparr==iter_merge)].sum(axis=0)
        # time_merge_ed = time.time()
        # print(f'time used for merging this onemask is {time_merge_ed-time_merge_st}')
        masks_merged = sp.vstack((masks_merged,sp.dok_matrix(mask_arr)), format='dok')
        # time_store_ed = time.time()
        # print(f'time used for storing this onemask is {time_store_ed-time_merge_ed}')

    return masks_merged







if __name__=='__main__':
    print('This is the garage for relation calculation code')
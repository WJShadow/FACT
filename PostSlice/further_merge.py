import numpy as np
import scipy.sparse as sp
from numba import njit, prange

def merge_IoU(masks: sp.csr_matrix, timestamps, thresh_IoU, thresh_refine):
    '''Using the IoU as the metric for merging masks segmented from different time windows
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **timestamps** list(ndarray): list of array of indices when a mask is present in the time line 
        **thresh_IoU** [float]: the threshold of IoU for merging

    Outputs:
    --------
        **masks_merged_IoU** sparse matrix MxP [int]: the masks after merging in sparse matrix
        **timestamps_merged_IoU** list of ndarray of indices when a mask is present in the time line 
    '''
    # refine mask boarders 
    masks = sp.vstack([msk >= msk.max() * thresh_refine for msk in masks]).astype('float') 

    # Calculation of IoU mat
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    overlap_mat = overlap_mat - np.diag(area_arr) 
    area_row = area_arr[:,np.newaxis].repeat(masks.shape[0], axis=1) 
    union_mat = area_row + area_row.T - overlap_mat 
    IoU_mat = overlap_mat / union_mat 
    # IoU_mat = IoU_mat - np.diag(IoU_mat.diagonal())

    # find merging pairs to process
    mask_num = masks.shape[0]
    group_situs = find_mask_groups(IoU_mat, thresh_IoU) # grouping masks according to the IoU values and threshold 
    keep_list = np.unique(group_situs)
    transform_mat = sp.csr_matrix((np.ones(mask_num), (group_situs, np.arange(mask_num))), (mask_num, mask_num))[keep_list]
    masks_merged_IoU = transform_mat.dot(masks)

    # merge timestamps of each individual mask 
    timestamps_merged_IoU = merge_timestamps(timestamps, group_situs, keep_list)

    return masks_merged_IoU, timestamps_merged_IoU



@njit('i4[:](f8[:,:],f4)', parallel= True, fastmath=True)
def find_mask_groups(metric_mat, thresh_metric):
    ''' Rapidly find groups where all masks within have metrics larger than then threshold 
    
    Inputs:
    -------
        **metric_mat** ndarray NxN: the matrix storing metric according to which masks will be grouped
        **thresh_metric** float: the threshold of metric for grouping masks

    Outputs:
    --------
        **groups** ndarray: array of indices indicating which mask each mask is grouped to

    
    '''
    N = metric_mat.shape[0]
    groups = -np.ones(N, dtype=np.int32)  # Initiate the signatures of all masks

    # find all paired masks according to the IoU values among them 
    for i in range(N):
        for j in range(i + 1, N):
            if metric_mat[i, j] > thresh_metric:
                # when no group of these two masks has been created
                if groups[i] == -1 and groups[j] == -1:
                    groups[i] = i
                    groups[j] = i
                # when one of the masks has been in one group
                elif groups[i] == -1:
                    groups[i] = groups[j]
                elif groups[j] == -1:
                    groups[j] = groups[i]
                # when two masks all have their own groups
                else:
                    group_to_merge = groups[j]
                    for k in range(N):
                        if groups[k] == group_to_merge:
                            groups[k] = groups[i]

    # when none of them has been assigned to any group
    for i in range(N):
        if groups[i] == -1: 
            groups[i] = i 
 
    return groups



def merge_timestamps(timestamps_orig, group_status, keep_list):
    ''' Merge timestamps of each individual mask
    
    Inputs:
    -------
        **timestamps_orig** list of ndarray: timestamps before merge 
        **group_status** ndarray: the targeted indice each timestamp is to be merged
        **keep_list** ndarray: indices of masks to be kept(either merged or independent)
    
    Outputs:
    --------
        **timestamps_merged** list of ndarray: timestamps after merge
    
    '''
    for iter_msk in prange(group_status.size):
        if not group_status[iter_msk]==iter_msk:
            targ_msk = group_status[iter_msk]
            ts_major = timestamps_orig[targ_msk]
            ts_new = timestamps_orig[iter_msk]
            timestamps_orig[targ_msk] = np.hstack((ts_major, ts_new))
    
    timestamps_merged = [np.unique(timestamps_orig[msk]) for msk in keep_list]

    return timestamps_merged



def merge_consume(masks: sp.csr_matrix, timestamps, thresh_consume, thresh_refine, max_area):
    '''Using the consume ratio as the metric for merging masks segmented from different time windows
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **timestamps** list(ndarray): list of array of indices when a mask is present in the time line 
        **thresh_consume** float: threshold IoU for merging
        **thresh_refine** float: threshold for refinning masks boarders

    Outputs:
    --------
        **masks_merged_consume** sparse matrix MxP [int]: the masks after merging in sparse matrix
        **timestamps_merged_consume** list of ndarray of indices when a mask is present in the time line 
    '''
    # refine mask boarders
    masks = sp.vstack([msk >= msk.max() * thresh_refine for msk in masks]).astype('float')

    # Calculation of IoU mat
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    overlap_mat = overlap_mat - np.diag(area_arr) 
    area_row = area_arr[:,np.newaxis].repeat(masks.shape[0], axis=1) 
    # union_mat = area_row + area_row.T - overlap_mat 
    consume_mat = overlap_mat / area_row
    # IoU_mat = IoU_mat - np.diag(IoU_mat.diagonal())

    # find merging pairs to process
    area_arr_cp = area_arr.copy()
    mask_num = masks.shape[0]
    group_situs = find_mask_groups_areafilt(consume_mat, thresh_consume, area_arr_cp, max_area) # grouping masks according to the consume values and threshold 

    keep_list = np.unique(group_situs)

    if keep_list[0]<-1:  # when paired masks larger than max_area exist, del them
        keep_list = keep_list[1:] 
        del_posis = (group_situs < -1).nonzero()[0]
        for iter_del in  del_posis: 
            masks[iter_del, masks[iter_del,:].nonzero()[1]] = 0 
        group_situs[del_posis] = 0 
    
    transform_mat = sp.csr_matrix((np.ones(mask_num), (group_situs, np.arange(mask_num))), (mask_num, mask_num))[keep_list]
    masks_merged_consume = transform_mat.dot(masks)

    # merge timestamps of each individual mask 
    timestamps_merged_consume = merge_timestamps(timestamps, group_situs, keep_list)

    return masks_merged_consume, timestamps_merged_consume



@njit('i4[:](f8[:,:],f8,f8[:],i8)', parallel= True, fastmath=True)
def find_mask_groups_areafilt(metric_mat, thresh_metric, area, thresh_area):
    ''' Rapidly find groups where all masks within have metrics larger than then threshold 
    And filter out masks with area larger than the area threshold
    
    Inputs:
    -------
        **metric_mat** ndarray NxN: the matrix storing metric according to which masks will be grouped
        **thresh_metric** float: the threshold of metric for grouping masks

    Outputs:
    --------
        **groups** ndarray: array of indices indicating which mask each mask is grouped to

    
    '''
    N = metric_mat.shape[0] 
    groups = -np.ones(N, dtype=np.int32)  # Initiate the signatures of all masks
    
 
    # find all paired masks according to the metric values among them 
    for i in range(N): 
        for j in range(N): 
            if metric_mat[i, j] >= thresh_metric: 
                if area[i]>area[j] and area[i]>thresh_area:
                    groups[i] = -2 
                    continue 
                if area[j]>area[i] and area[j]>thresh_area: 
                    groups[j] = -2 
                    continue 

                # when no group of these two masks has been created
                if groups[i] == -1 and groups[j] == -1: 
                    groups[i] = i 
                    groups[j] = i 
                # when one of the masks has been in one group
                elif groups[i] == -1: 
                    groups[i] = groups[j] 
                elif groups[j] == -1: 
                    groups[j] = groups[i] 
                # when two masks all have their own groups
                else:
                    group_to_merge = groups[j]
                    for k in range(N):
                        if groups[k] == group_to_merge:
                            groups[k] = groups[i]

    # when none of them has been assigned to any group
    for i in range(N): 
        if groups[i] == -1: 
            groups[i] = i 
    
    return groups
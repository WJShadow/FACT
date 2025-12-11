import numpy as np
import scipy.sparse as sp 


def filter_active(masks_to_filter, timestamps_to_filter, thresh_acti=1, thresh_refine=0.5):
    '''Filter masks according to the valid active frames 

    Inputs: 
        **masks_to_filter** sparse_matrix: masks to be filtered in sparse matrix
        **timestamps_to_filter** list(ndarray): timestamps of all masks
        **thresh_acti** int: threshold of frames a neuron is considered to be active 
        **thresh_refine** float: threshold to refine the boarder of masks

    Outputs:
        **masks_filtered** sparse_matrix: masks after filtering in sparse format
        **timestamps_filtered** list(ndarray): list of array stored 

    '''
    num_masks=masks_to_filter.shape[0]  
    if num_masks==0:
        print('No mask detected, modify given params to get more possible recognitions')

    if thresh_acti>1: # when filter time window set to more than one

        stats_filtered = np.zeros(num_masks, dtype='bool')
        for iter_msk in range(num_masks):
            times_diff1 = timestamps_to_filter[iter_msk][thresh_acti-1:] - timestamps_to_filter[iter_msk][:1-thresh_acti]
            # indicators of whether the neuron was active for "cons" frames
            stats_filtered[iter_msk] = np.any(times_diff1==thresh_acti-1) 

        masks_filtered = masks_to_filter[stats_filtered]
        # timestamps_filtered = timestamps_to_filter[stats_filtered]
        timestamps_filtered = [ts for (s,ts) in zip(stats_filtered,timestamps_to_filter) if s]
    else:
        masks_filtered = masks_to_filter
        timestamps_filtered = timestamps_to_filter
    
    masks_filtered = sp.vstack([msk >= msk.max() * thresh_refine for msk in masks_filtered],format='csr').astype(np.int64)

    return masks_filtered, timestamps_filtered


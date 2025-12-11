import numpy as np
import scipy.sparse as sp
import time

def olcancel(masks_init: sp.dok_matrix):
    ''' 
    Cancel the overlapping area in each mask
    
    Inputs:
    -------
        **masks_init** dok_matrix NxP [int]: all masks in dok_matrix

    '''
    overlap_mat =  masks_init.dot(masks_init.T).toarray()
    overlap_mat = overlap_mat - np.diag(overlap_mat.diagonal())  # delete the diagonal numbers for cancelling the self-overlap

    olstatus = overlap_mat.sum(axis=1)  
    list_olmasks = olstatus.nonzero()[0]  # locate the masks which have overlap area with other masks
    
    num_olmasks = list_olmasks.shape[0]
    clear_container = sp.dok_matrix((num_olmasks, masks_init.shape[1]))  # container for storing positions to be cleared in each mask with overlap
    
    lab_cont = 0
    for iter_olmask in list_olmasks:  # Iteratively calculate the overlap area of each pair of overlap mask
        # if (not overlap_mat[iter_olmask,:].sum()>0):  # skip when the current mask has been calculated all overlap area
        #     lab_cont += 1
        #     continue
        list_currentol = overlap_mat[iter_olmask,:].nonzero()[0]  # masks has overlapping area with current mask
        for iter_currentol in list_currentol:
            clear_container[lab_cont,:] += masks_init[iter_olmask,:].multiply(masks_init[iter_currentol,:])
            # clear_container[iter_currentol,:] += masks_init[iter_olmask,:].multiply(masks_init[iter_currentol,:])
            # overlap_mat[iter_olmask, iter_currentol], overlap_mat[iter_currentol, iter_olmask] = 0, 0  # label the two masks in the overlap mat

        lab_cont += 1
    
    print(clear_container.toarray())

    lab_cont = 0
    for iter_olmask in list_olmasks:  # Iteratively clear the overlapping area of each mask
        clearposis = clear_container[lab_cont,:].nonzero()[1]
        print(clearposis)
        masks_init[iter_olmask, clearposis] = 0
        lab_cont += 1
    
    return


def cancelol_corr(raw_vid:np.ndarray, masks_added_oneinitmask:sp.dok_matrix, olratio_mat:np.ndarray, olratio_thresh:np.float64):
    # **raw_vid** ndarray HxWxT: raw video
    (H,W,T) = raw_vid.shape
    processed_labmat = np.zeros(olratio_mat.shape)  # matrix for labeling all processed mask pairs
    targ_posis = np.where((olratio_mat>0) & (olratio_mat<olratio_thresh))  # locate all mask pairs to be processed
    
    time_store_all = 0
    for iter_m1, iter_m2 in zip(targ_posis[0],targ_posis[1]):  # Iteratively process each pair of masks
        if (not (processed_labmat[iter_m1, iter_m2]==0 and processed_labmat[iter_m2, iter_m1]==0)): # skip when this pair has been processed
            continue
        mask1_mat = masks_added_oneinitmask[iter_m1,:].reshape((H,W)).toarray()
        mask2_mat = masks_added_oneinitmask[iter_m2,:].reshape((H,W)).toarray()
        posis1 = mask1_mat.nonzero()
        posis2 = mask2_mat.nonzero()
        
        overlap_mat = (masks_added_oneinitmask[iter_m1,:].multiply(masks_added_oneinitmask[iter_m2,:])).reshape((H,W)).toarray()  # locate the overlapping pixels among these two masks
        posis_ol = overlap_mat.nonzero()
        
        mask1_mat[posis_ol] = 0  # keep only the non-overlapping pixels
        mask2_mat[posis_ol] = 0

        meantrace1 = raw_vid[posis1[0], posis1[1],:].mean(axis=0)
        meantrace2 = raw_vid[posis2[0], posis2[1],:].mean(axis=0)

        overlap_traces = raw_vid[posis_ol[0], posis_ol[1]]

        corr_mat = np.corrcoef(np.vstack((meantrace1, meantrace2, overlap_traces)),rowvar=True)
        corr_status = corr_mat[2:, 0:2]  # Nx2 correlation status
        
        targs = corr_status.argmax(axis=1)  # target mask of each overlapping pixel

        mask1_mat[posis_ol[0][np.where(targs==0)[0]], posis_ol[1][np.where(targs==0)[0]]] = 1
        mask2_mat[posis_ol[0][np.where(targs==1)[0]], posis_ol[1][np.where(targs==1)[0]]] = 1
        
        mask1_dok = sp.dok_matrix(mask1_mat.reshape((1,-1)))
        mask2_dok = sp.dok_matrix(mask2_mat.reshape((1,-1)))

        time_store_st = time.time()   
        
        masks_added_oneinitmask[iter_m1,masks_added_oneinitmask[iter_m1,:].nonzero()[1]] = 0  
        masks_added_oneinitmask[iter_m2,masks_added_oneinitmask[iter_m2,:].nonzero()[1]] = 0
        masks_added_oneinitmask[iter_m1,mask1_dok.nonzero()[1]] = mask1_dok[mask1_dok.nonzero()]
        masks_added_oneinitmask[iter_m2,mask2_dok.nonzero()[1]] = mask2_dok[mask2_dok.nonzero()]
        ## this is the main cause of the long processing time interval, already fixed
        # masks_added_oneinitmask[iter_m1,:] = mask1_dok  # update the masks in the mask dok matrix
        # masks_added_oneinitmask[iter_m2,:] = mask2_dok
        time_store_all += time.time() - time_store_st

        processed_labmat[iter_m1, iter_m2],processed_labmat[iter_m2, iter_m1] = 1,1  # label the mask pair as processed
        
    # print(f'time spent for storing masks in the olcancel function is {time_store_all}')

    return
    
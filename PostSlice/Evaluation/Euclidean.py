import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from math import sqrt
import numba
from numba import prange
from numba import jit
import time


# @jit(nopython=False,fastmath=True, parallel=True, cache=True)
def calc_COM(masks, H, W):
    '''Calculation of center of mass 
    
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

    COMs = np.zeros((N_mask,2), dtype=np.int32)  # array for storing the COM positions
    for iter_mask in prange(N_mask):
        mask_array = masks[iter_mask,:].toarray().squeeze()
        array_posi =  np.where(mask_array>0)
        num_posi = np.sum(mask_array>0)
        if num_posi == 0:
            sub_h = -1
            sub_w = -1
        else:
            # COM_ind = np.int32(round(np.float32(np.sum(array_posi)) / np.float32(num_posi)))  # calculating the linear ind position of COM
            sum_h = 0
            sum_w = 0
            for iter_posi in prange(num_posi):
                pix_ind = array_posi[0][iter_posi]
                h_pix = pix_ind // W
                w_pix = pix_ind - h_pix * W

                sum_h += h_pix
                sum_w += w_pix

            # sub_h = COM_ind // W  # converting the ind coordinate of COM position to sub coordinate:  (sub_h, sub_w)
            # sub_w = COM_ind - sub_h*W
            sub_h = np.int32(round(np.float32(sum_h) / np.float32(num_posi)))
            sub_w = np.int32(round(np.float32(sum_w) / np.float32(num_posi)))
        
        COMs[iter_mask,0] = sub_h
        COMs[iter_mask,1] = sub_w
    
    return COMs


def evaluation_Euclidean(infer_masks, GT_masks, image_shape, pix_dist, thresh_IoU, thresh_dist):
    '''Main code for matching infer and GT masks according to the IoU and Euclidean distance thresholds
    
    Inputs: 
    -------
        **infer_masks** dok_matrix NxP[int32]: sparse dok_matrix storing the N_infer masks in vector form, P is the number of pixels
        **GT_masks** dok_matrix MxP[int32]: sparse dok_matrix storing the N_GT masks in vector form , P is the number of pixels
        **image_shape** tuple (int64, int64): shape of original image in pixels
        **pix_dist** [float64]: the distance between two neighboring pixels in unit of micrometer
        **thresh_IoU** [float64]: threshold of the minimum IoU for a infer mask to be paired to a GT mask 
        **thresh_dist** [float64]: threshold of the maximnum COM distance for a infer mask to be paired to a GT mask 
    
    Outputs:
    --------
        **match_infer_list** 1 x N_infer [int64]: a vector of num indicating which GT mask a inferred mask is matched to, -1 means no match
        **Precision** [np.float16]: The precision of inferred masks compared to the GT masks
        **Recall** [np.float16]: The recall of inferred masks compared to the GT masks
        **F1** [np.float16]: The harmonic average of Precision and Recall 

    '''
    (H,W) = image_shape 
    N_infer = infer_masks.shape[0]
    N_GT = GT_masks.shape[0]

    IoU_mat = 1 - cdist(infer_masks.A, GT_masks.A, 'Jaccard')  # calculating the IoU value calling the cdist function from scipy.distance (Jaccard distance is just 1 - IoU)
    
    COM_infer = calc_COM(infer_masks, H, W)  # COM of inferred and GT masks
    COM_GT = calc_COM(GT_masks, H, W)
    
    # The following two lines were aabandoned because the output of COM generator has been changed to array already
    # COM_infer = np.asarray(COM_infer_list)  # Transform to array type for numba acceleration code usage 
    # COM_GT = np.asarray(COM_GT_list)        # Note: annotate these two lines when using dict_COM (numba accelerated COM calculation code), cause its return is already a numpy array

    Eucdist_mat = euclidean_distance(COM_infer, COM_GT, pix_dist)  # calling the euclidean_distance function to calculate distances between each pair of masks
    # print(Eucdist_mat)  # code for confirming the corectness of distance calculation
    
    # Using obtained IoU_mat and Eucdist_mat and corresponding thresholds to choose pairs
    match_infer_list = match_IoUEuc(IoU_mat, Eucdist_mat, thresh_IoU, thresh_dist)

    match_situ = match_infer_list >= 0
    match_num = np.sum(match_situ)

    Precision = match_num / N_infer
    Recall = match_num / N_GT
    F1 = 2*Precision*Recall / (Precision+Recall) 
    
    return [match_infer_list, Precision, Recall, F1]



@jit("f8[:,:](i4[:,:], i4[:,:], f8)", nopython=True, parallel=True, cache=True, fastmath=True)
def euclidean_distance(COM_infer, COM_GT, pix_dist):
    '''Calculating the Euclidean distance between two sets of masks

    Inputs:
    -------
        **COM_infer** Nx2 [int64]: N_infer pairs of COM coordinates of the inferred masks
        **COM_GT** Nx2 [int64]: N_GT pairs of COM coordinates of the GT masks
        **pix_dist** [float64]: distance between two neighbouring pixels, in unit of micrometer
    
    Outputs:
    --------
        **COMdist_mat** N_infer x N_GT [float64]: distances between each pair of infer&GT masks

    '''
    
    N_infer = COM_infer.shape[0]
    N_GT = COM_GT.shape[0]

    
    COMdist_mat = np.zeros((N_infer, N_GT), dtype=np.float64)
    for iter_infer in prange(N_infer):
        hi = COM_infer[iter_infer,0]
        wi = COM_infer[iter_infer,1]
        for iter_GT in prange(N_GT):
            dist = sqrt((hi - COM_GT[iter_GT,0])**2 + (wi - COM_GT[iter_GT,1])**2) * pix_dist
            COMdist_mat[iter_infer, iter_GT] = dist

    return COMdist_mat
    

@jit("i8[:](f8[:,:], f8[:,:], f8, f8)", nopython=True, cache=True, fastmath=True)
def match_IoUEuc(IoU_mat, Eucdist_mat, thresh_IoU, thresh_dist):
    '''Matching the infer masks to the GT masks with IoU>thresh_IoU & Eucdist<thresh_dist 
    
    Inputs:
    -------
        **IoU_mat** N_infer x N_GT [int64]: Matrix storing IoUs between each pair of infer and GT masks
        **Eucdist_mat** [int64]: Matrix storing the Euclidean distance between each pair of infer and GT masks
        **thresh_IoU** [float64]: threshold of the min IoU for matching the mask pair
        **thersh_dist** [float64]: threshold of the max Euclidean distance for matching the mask pair

    Outputs:
    --------
        **matcharry** 1 x N_infer [int64]: array of matched GT mask number of each inferred mask, -1 means no match
    '''
    if (IoU_mat.shape[0] != Eucdist_mat.shape[0]):
        raise ValueError('IoU and Eucdist matrix do not have the same number of infer masks')
    if (IoU_mat.shape[1] != Eucdist_mat.shape[1]):
        raise ValueError('IoU and Eucdist matrix do not have the same number of GT masks')
    
    N_infer = IoU_mat.shape[0]
    N_GT = IoU_mat.shape[1]

    matcharry_infer = np.ones(N_infer, dtype=np.int64) * (-1)
    status_GT = np.ones(N_GT, dtype=np.int64) * (-1)

    N_found = 1
    while(N_found > 0):
        N_found = 0
        for iter_infer in range(N_infer):
            if (matcharry_infer[iter_infer] >= 0):  # skip those infer masks that have been matched already
                continue
            
            max_IoU = 0
            max_IoU_ind = -1

            for iter_GT in range(N_GT):  
                if ((IoU_mat[iter_infer, iter_GT]>thresh_IoU) and (Eucdist_mat[iter_infer,iter_GT]<thresh_dist)):
                    if ( (IoU_mat[iter_infer, iter_GT]>max_IoU) and (status_GT[iter_GT]<0) ):
                        max_IoU = IoU_mat[iter_infer, iter_GT]
                        max_IoU_ind = iter_GT

            if (max_IoU_ind >= 0): # when at least one pair is found
                matcharry_infer[iter_infer] = max_IoU_ind
                N_found += 1
                
            else: # when no mask pair found, setting -1 marking this mismatch
                matcharry_infer[iter_infer] = -1
        
        if (N_found == 0):
            break

        # Code in the following section make sure that one GT mask can only have one corresponding infer mask to form a pair
        for iter_GT in range(N_GT):  # iteratively check if more than one infer mask correspond to one GT masks, if so choose only the one with highest IoU
            arr_status = (matcharry_infer==iter_GT) 
            num_match = np.sum(arr_status) 
            if (num_match>0):
                status_GT[iter_GT] = 1

            if (num_match>1): 
                posi_match = np.where(matcharry_infer==iter_GT) 
                max_IoU = 0 
                max_IoU_ind = -1 

                for iter_match in posi_match[0]: # searching for the pair with the highest IoU 
                    if (IoU_mat[iter_match, iter_GT]>max_IoU): 
                        max_IoU = IoU_mat[iter_match,iter_GT] 
                        max_IoU_ind = iter_match 

                for iter_match in posi_match[0]: # setting all other infer mask status to -1, leaving only the one with the highest IoU
                    if (iter_match==max_IoU_ind): 
                        matcharry_infer[iter_match] = iter_GT 
                    else: 
                        matcharry_infer[iter_match] = -1
        
    return matcharry_infer


            


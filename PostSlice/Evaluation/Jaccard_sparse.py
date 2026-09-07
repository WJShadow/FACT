import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def evaluation_jaccard(masks_infer, masks_GT, thresh_jacdist):
    # Calculation of IoU mat
    overlap_mat = masks_infer.dot(masks_GT.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    
    area_infer = (masks_infer.dot(masks_infer.T).toarray()).diagonal() 
    areamat_infer = np.repeat(np.expand_dims(area_infer.T,axis=1), masks_GT.shape[0], axis=1)
    area_GT = (masks_GT.dot(masks_GT.T).toarray()).diagonal()  
    areamat_GT = np.repeat(np.expand_dims(area_GT,axis=0), masks_infer.shape[0], axis=0)

    union_mat = areamat_infer + areamat_GT - overlap_mat 

    IoU_mat = np.divide(overlap_mat, union_mat, dtype=np.float64)
    # IoU_mat = overlap_mat.astype(np.float64) / union_mat.astype(np.float64)
    Jdist_mat = 1 - IoU_mat

    
    Jdist_mat[Jdist_mat > thresh_jacdist] = 2  # for pairs with Jaccard_distance>thresh_jacdist, consider them to be not matching
    row_match, col_match = linear_sum_assignment(Jdist_mat)  # calling the Hungarian algorithm to match the correspodning GT masks and the inferred masks
    match_list = Jdist_mat[row_match, col_match] < 1  # Excluding the mask pairs with a Jaccard distance higher thant the thresh_jacdist
    TP_num = np.sum(match_list) 

    match_GT_list = []  # creating a list of index num for GT masks identified
    for (iter_row,iter_col) in zip(row_match,col_match):
        if Jdist_mat[iter_row, iter_col] < 2:
            match_GT_list.append(iter_col)
    
    
    infer_num = masks_infer.shape[0]  # calculating the inferred positive num and GT positive num
    GT_num = masks_GT.shape[0]
    
    Precision = np.float64(TP_num) / np.float64(infer_num)  # calculating Precision and Recall value using the TP_num and positive num in GT and inferred masks
    Recall = np.float64(TP_num) / np.float64(GT_num)
    
    F1 = 2*Precision*Recall / (Precision+Recall)  # the harmonic average of precision and recall

    return [match_GT_list, Precision, Recall, F1]


def evaluation_jaccard_simple(masks_infer, masks_GT, thresh_jacdist):
    # Calculation of IoU mat
    overlap_mat = masks_infer.dot(masks_GT.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    
    area_infer = (masks_infer.dot(masks_infer.T).toarray()).diagonal() 
    areamat_infer = np.repeat(np.expand_dims(area_infer.T,axis=1), masks_GT.shape[0], axis=1)
    area_GT = (masks_GT.dot(masks_GT.T).toarray()).diagonal()  
    areamat_GT = np.repeat(np.expand_dims(area_GT,axis=0), masks_infer.shape[0], axis=0)

    union_mat = areamat_infer + areamat_GT - overlap_mat 

    IoU_mat = np.divide(overlap_mat, union_mat, dtype=np.float64)
    # IoU_mat = overlap_mat.astype(np.float64) / union_mat.astype(np.float64)
    Jdist_mat = 1 - IoU_mat

    
    Jdist_mat[Jdist_mat > thresh_jacdist] = 2  # for pairs with Jaccard_distance>thresh_jacdist, consider them to be not matching
    row_match, col_match = linear_sum_assignment(Jdist_mat)  # calling the Hungarian algorithm to match the correspodning GT masks and the inferred masks
    match_list = Jdist_mat[row_match, col_match] < 1  # Excluding the mask pairs with a Jaccard distance higher thant the thresh_jacdist
    TP_num = np.sum(match_list) 
    
    infer_num = masks_infer.shape[0]  # calculating the inferred positive num and GT positive num
    GT_num = masks_GT.shape[0]
    
    Precision = np.float64(TP_num) / np.float64(infer_num)  # calculating Precision and Recall value using the TP_num and positive num in GT and inferred masks
    Recall = np.float64(TP_num) / np.float64(GT_num)
    
    F1 = 2*Precision*Recall / (Precision+Recall)  # the harmonic average of precision and recall

    return [Precision, Recall, F1]
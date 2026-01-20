import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def evaluation_jaccard(infer_masks, GT_masks, thresh_jacdist):
    '''Evaluating the segmentation results according to the jaccard distance between masks
    Using the Jaccard distance to measure the distance between two masks from GT and inferred masks, then use the Hungarian algorithm to match mask pairs
    
    Inputs:
    -------
        **infer_masks** dok_matrix NxP[int32]: sparse dok_matrix storing the N_infer masks in vector form, P is the number of pixels
        **GT_masks** dok_matrix MxP[int32]: sparse dok_matrix storing the N_GT masks in vector form , P is the number of pixels
        **thresh_jacdist** [np.float16]: threshold of jaccard-distance for matching inferred masks and GT masks

    Outputs: [match_GT_list, Precision, Recall, F1]
    --------
        **match_GT_list** Nx1[bool]: a vector of flag indicating whether the GT mask has been matched by a inferred mask
        **Precision** [np.float16]: The precision of inferred masks compared to the GT masks
        **Recall** [np.float16]: The recall of inferred masks compared to the GT masks
        **F1** [np.float16]: The harmonic average of Precision and Recall 

        Note: The output of this function is in the format of a TriTuple data, which may cause incompatiility in the numba acceleration
    '''
    if infer_masks.shape[0]>1000:
        Jdist_mat = cdist(infer_masks.A, GT_masks.A, 'Jaccard')  # calling scipy.distance.cdist to calculate the Jaccard distance between the inferred masks and the GT masks
    else:
        Jdist_mat = fast_jaccard_cp(infer_masks, GT_masks)
    Jdist_mat[Jdist_mat > thresh_jacdist] = 2  # for pairs with Jaccard_distance>thresh_jacdist, consider them to be not matching
    
    row_match, col_match = linear_sum_assignment(Jdist_mat)  # calling the Hungarian algorithm to match the correspodning GT masks and the inferred masks
    match_list = Jdist_mat[row_match, col_match] < 1  # Excluding the mask pairs with a Jaccard distance higher thant the thresh_jacdist
    TP_num = np.sum(match_list) 

    match_GT_list = []  # creating a list of index num for GT masks identified
    for (iter_row,iter_col) in zip(row_match,col_match):
        if Jdist_mat[iter_row, iter_col] < 2:
            match_GT_list.append(iter_col)
    
    
    infer_num = infer_masks.shape[0]  # calculating the inferred positive num and GT positive num
    GT_num = GT_masks.shape[0]
    
    Precision = np.float32(TP_num) / np.float32(infer_num)  # calculating Precision and Recall value using the TP_num and positive num in GT and inferred masks
    Recall = np.float32(TP_num) / np.float32(GT_num)
    
    F1 = 2*Precision*Recall / (Precision+Recall)  # the harmonic average of precision and recall

    return [match_GT_list, Precision, Recall, F1]


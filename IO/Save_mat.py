# Code for writing the results masks into the mat file for further processing using Matlab 
import os
# from natsort import natsorted
import numpy as np
from datetime import datetime
import scipy
import pathlib

##                                               Storing code for the mat file using Matlab io functions
def savemask_mat(masks, maskshape_2d,  save_pth, annotate):
    '''
    Save all masks into the mat file for further processing using Matlab and other softwares capable of processing the mat file 
    Inputs:
    -------
        **masks** lil_matrix NxP: all the masks to be saved stored in the sparse_matrix format
        **maskshape_2d** tuple(int, int): shape of masks in the 2D h-w dimension
        **save_pth**: path_like format of the path for storing the output mat file, usually stored in the format compatible with the pathlib path format
    Outputs:
    --------
        Directly output the mat file with masks sotred within to the pth+"FACT_masks.mat" pth location with all the masks stored in the corresponding struct element with 
        their x and y positions stored individually and indipendently 

    Written 20231227, using the scipy.io library for directly storing the array data into the mat file 

    '''
    masks_posis = []
    for iter_mask in range(masks.shape[0]):
        mask_posis_ind = masks[iter_mask,:].nonzero()[1]
        mask_posis_sub = np.unravel_index(mask_posis_ind, maskshape_2d)  # unravel the index using unravel_index func in numpy 
        posis_dict = {"x": mask_posis_sub[0], "y":mask_posis_sub[1]}
        # print(posis_dict)  # uncomment for debugging
        masks_posis.append(posis_dict)
    
    masks_posis = np.array(masks_posis, dtype=object)
    
    datetime_now = datetime.now()  # obtain the date and time to store together with the masks
    timestamp = datetime_now.strftime('%c')
    
    scipy.io.savemat( 
        file_name=save_pth+'_'+annotate+'.mat', mdict={
            "datetime": timestamp,
            "masks": masks_posis
        })


def savemask_mat_withtrace(masks, maskshape_2d, traces, save_pth, maskfrom):
    '''
    Save all masks into the mat file for further processing using Matlab and other softwares capable of processing the mat file 
    Inputs:
    -------
        **masks** lil_matrix NxP: all the masks to be saved stored in the sparse_matrix format
        **maskshape_2d** tuple(int, int): shape of masks in the 2D h-w dimension
        **traces** ndarray NxT: time traces of all masks
        **save_pth**: path_like format of the path for storing the output mat file, usually stored in the format compatible with the pathlib path format
    Outputs:
    --------
        Directly output the mat file with masks sotred within to the pth+"FACT_masks.mat" pth location with all the masks stored in the corresponding struct element with 
        their x and y positions stored individually and indipendently 

    Using the scipy.io library for directly storing the array data into the mat file 

    '''
    masks_posis = []
    for iter_mask in range(masks.shape[0]):
        mask_posis_ind = masks[iter_mask,:].nonzero()[1]
        mask_posis_sub = np.unravel_index(mask_posis_ind, maskshape_2d)  # unravel the index using unravel_index func in numpy 
        posis_dict = {"x": mask_posis_sub[0], "y":mask_posis_sub[1]}
        # print(posis_dict)  # uncomment for debugging
        masks_posis.append(posis_dict)
    
    masks_posis = np.array(masks_posis, dtype=object)
    
    datetime_now = datetime.now()  # obtain the date and time to store together with the masks
    timestamp = datetime_now.strftime('%c')
    
    scipy.io.savemat( 
        file_name=save_pth+'_'+maskfrom+'_masks_trace.mat', mdict={
            "datetime": timestamp,
            "masks": masks_posis,
            "traces": traces
        })

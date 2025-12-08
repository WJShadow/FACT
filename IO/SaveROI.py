import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import roifile
import cv2
from roifile import ImagejRoi, ROI_TYPE,ROI_OPTIONS,roiwrite
# from ReadNFMGT import read_NF_ManualGT



def save_roi_imagej(masks, maskshape_2d, save_path):
    '''
    Code for saving the infrred masks into a zipped file containing multiple roi files 
    For direct read-in using ImageJ

    Inputs:
    -------
        **masks** sparse_matrix NxP: matrix storing all the inferred masks in one sparse matrix for saving memory space 
        **maskshape_2d** (np.int64, np.int64): 2-dimensional shape of the inferred masks 
        **save_path** pathlike str: full path to the zip file, where the masks will be stored
    
    '''
    
    num_masks = masks.shape[0]
    list_rois = [] 
    for single_mask_arr in masks:
        single_mask_mat = single_mask_arr.toarray().reshape(maskshape_2d)
        single_mask_mat = single_mask_mat.astype(np.uint8)

        contours, _ = cv2.findContours(single_mask_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        boundary_points = []
        for contour in contours:
            for point in contour:
                x, y = point[0]
                boundary_points.append([x, y])  # 将坐标点改为列表格式
        boundary_points_array = np.array(boundary_points)

        # single_mask_boarder = np.zeros(maskshape_2d, dtype=np.uint8)
        # single_mask_boarder[boundary_points_array[:,1], boundary_points_array[:,0]] = 1

        single_roi = ImagejRoi.frompoints(boundary_points_array)
        single_roi.roitype = ROI_TYPE.FREEHAND

        list_rois.append(single_roi) 

    roiwrite(save_path, list_rois) 
    return

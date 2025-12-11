import numpy as np
import scipy.sparse as sp
import cv2
from multiprocessing import Pool


from PostSlice.com.feature import vhradio_sp

def openclose(masks: sp.csr_matrix, timestamps, thresh_area_min, thresh_area_max, img_shape_2d , kernel_size_open, kernel_size_close):
    '''Perform the open & close operation to masks with size greater than thresh_area
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **timestamps** list(ndarray): list of array of indices when a mask is present in the time line 
        **thresh_area** [float]: the threshold of area for performing morphology operation

    Outputs:
    --------
        **masks_operated** sparse matrix MxP [int]: the masks after operation in sparse matrix
    '''

    # Calculation of mask area
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    keep_list = []  # list storing masks to be kept
    operate_list = [] # list storing masks to be performed operation


    for mask_idx in range(masks.shape[0]):  # label all masks to be kept or operated
        if area_arr[mask_idx]>thresh_area_min and area_arr[mask_idx]<thresh_area_max:
            operate_list.append(mask_idx)
        else:
            keep_list.append(mask_idx)

    if len(operate_list)>0:
        masks_tooperate = sp.vstack([masks[idx] for idx in operate_list], format='csr')
        include_flag, masks_operated = operate_openclose(masks_tooperate, kernel_size_open, kernel_size_close, img_shape_2d)

        if len(keep_list)>0:
            masks_return = sp.vstack([masks[idx] for idx in keep_list], format='csr')
            if include_flag:
                masks_return = sp.vstack([masks_return, masks_operated], format='csr')
        else:
            masks_return = masks_operated

    else:
        masks_return = masks

    
    

    return masks_return


def operate_openclose(masks: sp.csr_matrix, kernel_size_open,kernel_size_close, img_shape_2d):
    '''
    Function for operating open-close morphology operations
    
    '''
    return_masks_list = []  
    include_flag = 0
    for msk in masks:
        msk_mat_arr = msk.reshape(img_shape_2d).toarray().astype(np.uint8)
        datatype = masks.dtype

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_open, kernel_size_open))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))

        closed_mask = cv2.morphologyEx(msk_mat_arr, cv2.MORPH_CLOSE, kernel_close)  

        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open)  
    

        num_segmasks, segmasks_im = cv2.connectedComponents(opened_mask)  
    

        
        for i in range(1, num_segmasks):  

            component_mask = (segmasks_im == i).astype(np.uint8)  

            closed_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel_close)  
            closed_mask_sp = sp.csr_matrix(closed_mask)  
            return_masks_list.append(closed_mask_sp.reshape((1,-1)).astype(datatype))  
        
    if len(return_masks_list)>0:
        return_masks = sp.vstack([mskr for mskr in return_masks_list], format='csr')
        include_flag = 1
    else:
        return_masks = []
        include_flag = 0
    return include_flag, return_masks


def ratiofilter(masks: sp.csr_matrix, thresh_hwratio, img_shape_2d):
    '''Filter out masks with a h-w ratio greater than thresh_hwratio of less than 1/thresh_hwratio
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **thresh_hwratio** [float]: threshold of shape ratio
    Outputs:
    --------
        **masks_filtered** sparse matrix MxP [int]: the masks after operation in sparse matrix
    '''

    vhrads, vhmaxmin = vhradio_sp(masks, img_shape_2d)
    keep_list = []

    for idx in range(masks.shape[0]):
        if (vhrads[idx,0]/vhrads[idx,1]>thresh_hwratio) or (vhrads[idx,1]/vhrads[idx,0]>thresh_hwratio):
            continue 
        else:
            keep_list.append(idx)

    return_masks = sp.vstack([masks[idxr] for idxr in keep_list], format='csr')

    return return_masks




  
def is_ellipse_or_square(contour, i_mask, params):  
    perimeter = cv2.arcLength(contour, closed=True)  
    area = cv2.contourArea(contour)  

    if perimeter==0:
        print(f'perimeter of {i_mask} mask is 0')
        raise ValueError(f'Zero value encountoured in mask {i_mask}, check info below')
    if area==0:
        return False
        # print(f'area of {i_mask} mask is 0')   
        # raise ValueError(f'Zero value encountoured in mask {i_mask}, check info below')
      
    rect = cv2.minAreaRect(contour)  
    box = cv2.boxPoints(rect)  
    box = np.int0(box)  
    width = np.linalg.norm(box[0] - box[1])  
    height = np.linalg.norm(box[1] - box[2])  
      

    form_factor = 4 * np.pi * area / (perimeter ** 2)  

    aspect_ratio = max(width, height) / min(width, height)  
      

    if params[0] < form_factor < params[1] and params[2] < aspect_ratio < params[3]:
    # if 0.2 < form_factor < 5 and 0.5 < aspect_ratio < 2:  
        return True  
    return False  
  
def filter_masks_ellipse(masks, mask_shape_2d, params):  
    filtered_masks = []  
    for i_mask, mask_sp in enumerate(masks):  
        mask = mask_sp.astype(np.uint8).reshape(mask_shape_2d).A

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
          
        filtered_contour = []  
        for contour in contours:  
            if len(contour)<2:
                continue
            if is_ellipse_or_square(contour, i_mask, params):  
                filtered_contour.append(contour)  

        if len(filtered_contour) == 0:
            continue
          
        new_mask = np.zeros_like(mask)  
        new_mask = cv2.drawContours(new_mask, filtered_contour, -1, (1), thickness=cv2.FILLED)  
        new_mask = sp.csr_matrix(new_mask.reshape(1,-1), dtype=np.int64)
        filtered_masks.append(new_mask)  
    
    filtered_masks_sp = sp.vstack([msk for msk in filtered_masks], format='csr')
    return filtered_masks_sp

def process_single_mask(args):
    i_mask, mask_sp, mask_shape_2d, params = args
    try:
        mask = mask_sp.astype(np.uint8).reshape(mask_shape_2d).A
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            if len(contour) < 2:
                continue
            if is_ellipse_or_square(contour, i_mask, params):
                filtered_contours.append(contour)
        if not filtered_contours:
            return None
        
        new_mask = np.zeros_like(mask)
        cv2.drawContours(new_mask, filtered_contours, -1, 1, cv2.FILLED)
        return sp.csr_matrix(new_mask.reshape(1, -1), dtype=np.int64)
    except Exception as e:
        print(f"Error processing mask {i_mask}: {e}")
        return None

def filter_masks_ellipse_p(masks, mask_shape_2d, params, n_processes=None):
    
    args_list = [(i, mask, mask_shape_2d, params) for i, mask in enumerate(masks)]
    
    
    with Pool(processes=n_processes) as pool:
        results = pool.imap(process_single_mask, args_list)
        filtered_masks = [res for res in results if res is not None]
    
    
    if not filtered_masks:
        return sp.csr_matrix((0, mask_shape_2d[0] * mask_shape_2d[1]), dtype=np.int64)
    return sp.vstack(filtered_masks, format='csr')



  

def enlarge(masks: sp.csr_matrix,thresh_area_min, thresh_area_max, img_shape_2d , kernel_size_dilation):
    return_masks_list = []  
    for msk in masks:
        msk_mat_arr = msk.reshape(img_shape_2d).toarray().astype(np.uint8)
        datatype = masks.dtype

        kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_dilation, kernel_size_dilation))

        enlarged_mask = cv2.morphologyEx(msk_mat_arr, cv2.MORPH_DILATE, kernel_dilation)  
        
        closed_mask_sp = sp.csr_matrix(enlarged_mask)  
        return_masks_list.append(closed_mask_sp.reshape((1,-1)).astype(datatype))  
        
    return_masks = sp.vstack([mskr for mskr in return_masks_list], format='csr')
    return return_masks


def enlarge_control(masks: sp.csr_matrix,thresh_area_min, thresh_area_max, img_shape_2d , kernel_size_dilation):
    return_masks_list = []  
    for msk in masks:
        msk_mat_arr = msk.reshape(img_shape_2d).toarray().astype(np.uint8)
        msk_area = np.sum(msk_mat_arr)
        datatype = masks.dtype
        if thresh_area_max>msk_area>thresh_area_min:
            kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_dilation, kernel_size_dilation))

            enlarged_mask = cv2.morphologyEx(msk_mat_arr, cv2.MORPH_DILATE, kernel_dilation)  
            
            closed_mask_sp = sp.csr_matrix(enlarged_mask)  
            return_masks_list.append(closed_mask_sp.reshape((1,-1)).astype(datatype))  
        else:
            return_masks_list.append(msk)
        
    return_masks = sp.vstack([mskr for mskr in return_masks_list], format='csr')
    return return_masks

def filter_masks_ellipse_allout(masks, mask_shape_2d, params):  
    list_normal_masks = []  
    list_special_masks = []
    for i_mask, mask_sp in enumerate(masks):  
        mask = mask_sp.astype(np.uint8).reshape(mask_shape_2d).A
        # 找到轮廓  
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
        # print(contours)
          
        len_contour_max = 0
        idx_contour = -1
        for idx, contour in enumerate(contours):  
            if len(contour)<2:
                continue
            if len(contour)>len_contour_max:
                len_contour_max = len(contour)
                idx_contour = idx
                
        if idx_contour<0:
            continue
        
        status = is_ellipse_or_square(contour, i_mask, params)
        if status:
            list_normal_masks.append(i_mask)
        else:
            list_special_masks.append(i_mask)
    
    normal_masks_sp = sp.vstack([masks[idx] for idx in list_normal_masks], format='csr')
    special_masks_sp = sp.vstack([masks[idx] for idx in list_special_masks], format='csr')

    return normal_masks_sp, special_masks_sp




def filter_small(masks: sp.csr_matrix, thresh_area_min, img_shape_2d):  ### 
    '''Filter out masks smaller than a set area
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **thresh_area_min** [float]: the threshold of area for filtering
        

    Outputs:
    --------
        **masks_operated** sparse matrix MxP [int]: the masks after operation in sparse matrix
    '''

    # Calculation of mask area
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    keep_list = []  # list storing masks to be kept
    operate_list = [] # list storing masks to be performed operation

    
    for mask_idx in range(masks.shape[0]):  # label all masks to be kept or operated
        if area_arr[mask_idx]>thresh_area_min:
            keep_list.append(mask_idx)

    masks_return = sp.vstack([masks[idx] for idx in keep_list], format='csr')

    return masks_return

def filter_small_fix(masks: sp.csr_matrix, thresh_area_min, img_shape_2d):
    '''Filter out masks smaller than a set area
    
    Inputs:
    -------
        **masks** dok_matrix NxP [int]: all masks to be considered merging in dok_matrix form
        **thresh_area_min** [float]: the threshold of area for filtering
        

    Outputs:
    --------
        **masks_operated** sparse matrix MxP [int]: the masks after operation in sparse matrix
    '''

    # Calculation of mask area
    overlap_mat = masks.dot(masks.T).toarray()  # matrix of overlapping area betweeen each pair of masks
    area_arr = overlap_mat.diagonal()  # area of each individual mask in paramater masks
    keep_list = []  # list storing masks to be kept
    operate_list = [] # list storing masks to be performed operation

    squ_thresh_area_min = thresh_area_min * thresh_area_min
    for mask_idx in range(masks.shape[0]):  # label all masks to be kept or operated
        if area_arr[mask_idx]>squ_thresh_area_min:
            keep_list.append(mask_idx)

    masks_return = sp.vstack([masks[idx] for idx in keep_list], format='csr')

    return masks_return


def remove_overlap(masks_in):
    """
    """

    masks = masks_in.tocsr()
    typ=masks_in.dtype
    N, P = masks.shape
    

    areas = np.array(masks.sum(axis=1)).flatten()
    

    # sorted_indices = np.argsort(areas)[::-1]
    sorted_indices = np.argsort(areas)
    

    result = sp.csr_matrix((N, P), dtype=bool)
    

    for i in sorted_indices:
        current_mask = masks[i].toarray().flatten().astype(bool)
        

        if not np.any(current_mask):
            continue
            

        existing_masks = result.sum(axis=0).A.flatten().astype(bool)
        

        overlap = current_mask & existing_masks

        cleaned_mask = current_mask & (~overlap)
        

        result[i] = cleaned_mask.astype(typ)
    
    return result
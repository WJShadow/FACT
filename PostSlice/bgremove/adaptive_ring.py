## Integrated ARBS in FACT

import numpy as np
import scipy.sparse as sp
from skimage import morphology, measure
import cv2
import numpy as np
import scipy.sparse as sp
from skimage import morphology, measure
import cv2

def adaptive_ring_bg_remove(masks, img_shape_2d, input_vid):

    ''' Remove background fluctuation using the adaptive-annualar background removal methods 
    masks: all masks segmented 
    img_shape_2d : the spatial shape of input_vid
    input_vid: TxHxW inp
    
    
    '''
    n, p = masks.shape
    (H, W) = img_shape_2d

    
    exclusion_zones = [] # not shown in the final results
    background_rings = [] # in the final outputs 

    for i in range(n):
        mask = masks[i].toarray().reshape(H, W)
        
        
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)
        
        
        if props:
            ellipse = props[0]
            major_axis_length = ellipse.major_axis_length
            minor_axis_length = ellipse.minor_axis_length
            
            
            dilation_radius_exclusion = int((major_axis_length + minor_axis_length) / 8)  # 1/6
            if dilation_radius_exclusion<1:
                dilation_radius_exclusion = 1
            se_exclusion = morphology.disk(dilation_radius_exclusion)
            exclusion_zone = morphology.binary_dilation(mask, se_exclusion)
            exclusion_zones.append(exclusion_zone)

            
            dilation_radius_background = int((major_axis_length + minor_axis_length) / 6)  # 1/4
            if dilation_radius_background< 2.5:
                dilation_radius_background= 2.5
            se_background = morphology.disk(dilation_radius_background)
            background_ring = morphology.binary_dilation(mask, se_background)
            background_ring = background_ring & ~exclusion_zone  # 减去禁区
            background_rings.append(background_ring)

   
    exclusion_zone_collection = np.max(np.array(exclusion_zones), axis=0).astype(np.uint8)

    
    for i in range(n):
        background_rings[i] = background_rings[i] & ~exclusion_zone_collection

    
    background_ring_sparse = sp.csr_matrix(np.array(background_rings).reshape(n, -1))


    areas_FACT = masks.astype(np.int64).dot(masks.astype(np.int64).T).diagonal()
    areas_bg_FACT = background_ring_sparse.dot(background_ring_sparse.T).diagonal()
        
    areas_current = areas_FACT
    areas_bg_current = areas_bg_FACT    
    current_masks = masks
    current_bg_masks = background_ring_sparse  

    traces_current = np.zeros((current_masks.shape[0], input_vid.shape[0]))
    traces_bg_current = np.zeros((current_masks.shape[0], input_vid.shape[0]))

    print(areas_current)
    print(areas_bg_FACT)


    for iter_fr in range(input_vid.shape[0]):
        value_fr = current_masks.dot(input_vid[iter_fr,:,:].reshape(1,-1).T)
        value_fr = np.divide(value_fr.squeeze(), areas_current)

        value_bg_fr = current_bg_masks.dot(input_vid[iter_fr,:,:].reshape(1,-1).T)
        value_bg_fr = np.divide(value_bg_fr.squeeze(), areas_bg_current)
        
        traces_current[:, iter_fr] = value_fr
        traces_bg_current[:, iter_fr] = value_bg_fr

    traces_bg_rm = traces_current-traces_bg_current*0.7

    traces_results = {
        'traces_cell': traces_current,
        'traces_bg': traces_bg_current,
        'traces_bg_rm': traces_bg_rm
    }

    return traces_results
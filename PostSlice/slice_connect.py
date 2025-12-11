import numpy as np
import scipy.sparse as sp
import cv2
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from numba import njit, types, prange

def slice_connect(infer_3d:np.ndarray, min_area:np.int32, max_area:np.int32, num_workers=0):
    '''Process each time slice frame in parallel to locate possible masks 
    
    Inputs:
    -------
        **infer_3d** ndarray TxHxW: The binary inference from model
        **min_area** : Threshold of the minimum pixels one mask should have 
        **max_area** : Masks with pixels larger than this threshold will be considered less important when merging masks
        **num_workers** : number of workers in the Threadpool 
    
    Outputs:
    --------
        **masks** sparse matrix nxP: all segmented masks
        **COMs** ndarray n: centers of corresponding segmented masks
        **areas** ndarray n: pixels of corresponding segmenetd masks
        **timestamps**: time steps of the corresponding masks 
    '''
    
    if (num_workers>1):  # using threadpool to accelerate segmentation
        # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            results = list(executor.map(process_slice, infer_3d, [min_area]*infer_3d.shape[0], [max_area]*infer_3d.shape[0]))  
    else:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(process_slice, infer_3d, [min_area]*infer_3d.shape[0], [max_area]*infer_3d.shape[0]))
        
    masks = sp.vstack([re[0] for re in results])
    COMs = np.vstack([re[1] for re in results])
    areas = np.hstack([re[2] for re in results])
    timestamps = np.hstack([mask_num * np.ones(re[0].shape[0], dtype='uint32') for (mask_num, re) in enumerate(results)])
    return masks, COMs, areas, timestamps




    
# Segment connected regions within each time slice of inferred image
def process_slice(slice: np.ndarray, min_area=0, max_area=200): 
    '''Rapidly segment connected regions within each time slice of inferred image
    
    Inputs:
    -------
        **slice** ndarray HxW: The sliced image to search for masks on
        **min_area** : Threshold of the minimum pixels one mask should have 
        **max_area** : Masks with pixels larger than this threshold will be considered less important when merging masks
    
    Outputs:
    --------
        **masks** sparse matrix nxP: all segmented masks in this sliced image
        **COMs** ndarray n: centers of corresponding segmented masks in this sliced image
        **areas** ndarray n: pixels of corresponding segmenetd masks in this sliced image
    
    '''
    # labels, centroids = connected_components(slice)
    # labels = mahotas.label(slice) # using the mahotas
    # labels = sitk.ConnectedComponent(slice)
    # labels = measure.label(slice,connectivity=1)
    # num_labels, labels, stats, COMs = cv2.connectedComponentsWithStatsWithAlgorithm(image=slice, connectivity=4, ccltype=cv2.CCL_SAUF, ltype=2) 
    num_labels, labels, stats, COMs = cv2.connectedComponentsWithStatsWithAlgorithm(image=slice, connectivity=4, ccltype=cv2.CCL_SAUF, ltype=4)
    keep_arr = (stats[:,4] > min_area).nonzero()[0]
    if keep_arr.size-1 > 0:
        sp_info = area_filter(stats=stats, labels=labels, keep_arr=keep_arr, max_area=max_area)
        masks = sp.csr_matrix((sp_info[2], (sp_info[0], sp_info[1])), shape=(keep_arr.size-1, labels.shape[0]*labels.shape[1]))  
        COMs = np.array(COMs[keep_arr[1:]]).astype(np.float32)
        areas = np.array(stats[keep_arr[1:],4])  
    else:
        masks = sp.csr_matrix((0, labels.shape[0] * labels.shape[1]))   
        COMs = np.empty((0, 2))
        areas = np.array([], dtype='int')

    return masks, COMs, areas





# @njit("i4[:,:](i4[:,:], u2[:,:], i8[:], i4)", parallel=True, fastmath=True, nogil=True, cache=True)
@njit("i4[:,:](i4[:,:], i4[:,:], i8[:], i4)", parallel=True, fastmath=True, nogil=True, cache=True)
def area_filter(stats, labels, keep_arr, max_area):
    '''Rapidly filter masks with area smaller than min_area while keeping masks with reasonable area

    Inputs:
    -------
        **stats** ndarray nx5 [int32]: stats from cv2.connectedComponentsWithStatsWithAlgorithm outputs  
        **labels** ndarray HxW [uint16]: labels from cv2.connectedComponentsWithStatsWithAlgorithm outputs  
        **keep_arr** ndarray m [int32]: list of neurons to keep(with area larger than min_area)
        **max_area** [int32]: masks with area larger than max_area will be given smaller weight
    
    Outputs:
    --------
        **sp_info** ndarray 3xm [int32]: row,col,value info for generating a sparse matrix that stores all masks in this frame 

    '''
    num_posis = stats[keep_arr[1:],4].sum()

    sp_info = np.zeros((3,num_posis), dtype=np.int32)
    img_shape = labels.shape

    
    for iter_mask in prange(keep_arr.size-1):
        iter_pix = stats[keep_arr[1:iter_mask+1],4].sum()
        lab = keep_arr[iter_mask+1]  # the current label to process
        val = 1 if stats[lab,4] < max_area else 1  # give masks with area larger than max_area less weight

        for h in range(stats[lab,1], stats[lab,1]+stats[lab,3]):
            for w in range(stats[lab,0], stats[lab,0]+stats[lab,2]):
                if (labels[h,w]==lab):
                    sp_info[0,iter_pix] = iter_mask
                    sp_info[1,iter_pix] = h * img_shape[1] + w
                    sp_info[2,iter_pix] = val
                    iter_pix += 1

    return sp_info



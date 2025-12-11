import numpy as np
import scipy.io
import scipy.sparse as sp
from scipy.sparse import lil_matrix

def Evaluate_STA(mat_path, npz_path):
    # Step 1: Process ideal_signalonly
    
    mat_data = scipy.io.loadmat(mat_path)

    ideal_signalonly = mat_data['ideal_signalonly'].astype(np.int64)
    H, W, N = ideal_signalonly.shape
    P = H * W
    
    GT_Neur_masks = lil_matrix((N, P), dtype=np.int64)
    for i in range(N):
        mask = ideal_signalonly[:, :, i]  # Transpose to HxW
        GT_Neur_masks[i] = mask.reshape(1, -1)
    
    pulse_cell = mat_data['pulse_neuron_cell']

    T = pulse_cell[0, 0].shape[1]  # Get T from first cell
    GT_Neur_time = np.zeros((N, T), dtype=bool)
    for i in range(N):
        GT_Neur_time[i] = pulse_cell[i, 0].astype(bool)
    
    npz_data = np.load(npz_path, allow_pickle=True)
    infer_mask3d = npz_data['infer_mask3d']
    
    H_inf, W_inf, T_inf = infer_mask3d.shape
    assert H == H_inf and W == W_inf, "Dimensions mismatch between mat and npz files"
    
    # Reshape to TxP
    infer_masks_time = lil_matrix((T, P), dtype=infer_mask3d.dtype)
    for t in range(T):
        frame = infer_mask3d[:, :, t]  # Transpose to HxW
        infer_masks_time[t] = frame.reshape(1, -1)
    
    TS_overlap = GT_Neur_masks.dot(infer_masks_time.transpose()).toarray()
    
    mask_areas = np.array(GT_Neur_masks.sum(axis=1)).flatten()
    TS_ratio = TS_overlap / mask_areas[:, np.newaxis]
    
    TS_infer_bw = (TS_ratio >= 0.2).astype(bool)
    TS_infer_bw_on = (TS_ratio >= 0.2).astype(bool)
    TS_infer_bw_off = (TS_ratio <= 1-0.2).astype(bool)
    TS_eva_onlysig_on = np.logical_and(TS_infer_bw_on, GT_Neur_time)
    TS_eva_onlysig_off = np.logical_and(TS_infer_bw_off, ~GT_Neur_time)
    
    STA_all = (TS_eva_onlysig_on.sum(axis=1)+TS_eva_onlysig_off.sum(axis=1)) / T
    
    return {
        'GT_Neur_masks': GT_Neur_masks,
        'GT_Neur_time': GT_Neur_time,
        'infer_masks_time': infer_masks_time,
        'STA_all' : STA_all
    }

import os
import numpy as np
import SimpleITK as sitk

def load_nii_inference(filepath, rotation_k=0, t_last=True):
    ''' Load inference result from nii.gz file
        Assuming the dims to be HxWxT
    
    Inputs:
    -------
        **filepath** [path]: path leading to the nii.gz inference file
        **rotation_k** [int(0-3)]: clockwise rotation degree (x90) to the inference

    Outputs:
    --------
        **[infer, infer_shape]** tuple[ndarray, _shape]: inference and its shape
    '''
    infer_nii = sitk.ReadImage(filepath)
    infer = sitk.GetArrayFromImage(infer_nii)
    if(t_last):  # Inputs of the FACT_Net were transposed to be [t,h,w], therefore transpose back if t_last==True
        infer=np.transpose(infer, [1,2,0])
    if (rotation_k != 0):
        np.rot90(infer, rotation_k, [0,1])
    
    infer_shape = infer.shape
    return infer, infer_shape
    
    
    # print("img shape:",img.shape)
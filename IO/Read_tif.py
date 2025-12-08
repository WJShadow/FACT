import numpy as np 
import scipy.sparse as sp
from tifffile import TiffFile
import SimpleITK as sitk

from ScanImageTiffReader import ScanImageTiffReader

def load_tiff(filepath):
    '''
    Loading tiff file and transform into the ndarray data format for futher processing
    
    Inputs:
    -------
        **filepath** pathlike: Fullpath to the tiff file 
    
    Outputs:
    -------
        **raw_vid_ndarr** ndarray TxHxW: The raw video loaded from the tiff file
    
    Exception:
        If the exception is raised with error printed as the algorithm is not working during the loading procedure, please check the path given and the tiff file carefully in case of 
        damaged file or incorrect filepath
    '''
    try:
        vidfile = sitk.ReadImage(filepath)
        raw_vid_ndarr = sitk.GetArrayFromImage(vidfile)
        if (len(raw_vid_ndarr.shape) < 3):
            print("Target file seems to be just a picture, please verify the path given")
            success_load = False
            # return
        else:
            success_load = True
    except:
        print("Failed with Sitk reader, trying ScanImageTiffReader")
        try:
            with ScanImageTiffReader(filepath) as raw_tif:
                if (len(raw_tif.shape()) < 3):
                    print("Target file seems to be just a picture, please verify the path given")
                    success_load = False
                    # return
                else:
                    raw_vid_ndarr = raw_tif.data()
                    success_load = True
        except:
            print("Failed with ScanImage, trying other methods instead for reading tiff file")
            success_load = False

    if not success_load: # If the ScanImageTiffReader is not working, try using the Tifffile instead
        try:
            raw_tif = TiffFile(filepath)
            raw_vid_ndarr = raw_tif.asarray()
            if (len(raw_tif.shape()) < 3):
                    print("Target file seems to be just a picture, please verify the path given")
                    success_load = False
                    return
        except:
            print("Failed to load the specified tiff file, please check the path and tiff file")
    return raw_vid_ndarr.astype(np.float32)



def load_tiff_16(filepath):
    '''
    Loading tiff file and transform into the ndarray data format for futher processing
    
    Inputs:
    -------
        **filepath** pathlike: Fullpath to the tiff file 
    
    Outputs:
    -------
        **raw_vid_ndarr** ndarray TxHxW: The raw video loaded from the tiff file
    
    Exception:
        If the exception is raised with error printed as the algorithm is not working during the loading procedure, please check the path given and the tiff file carefully in case of 
        damaged file or incorrect filepath
    '''
    try:
        vidfile = sitk.ReadImage(filepath)
        raw_vid_ndarr = sitk.GetArrayFromImage(vidfile)
        if (len(raw_vid_ndarr.shape) < 3):
            print("Target file seems to be just a picture, please verify the path given")
            success_load = False
            # return
        else:
            success_load = True
    except:
        print("Failed with Sitk reader, trying ScanImageTiffReader")
        try:
            with ScanImageTiffReader(filepath) as raw_tif:
                if (len(raw_tif.shape()) < 3):
                    print("Target file seems to be just a picture, please verify the path given")
                    success_load = False
                    # return
                else:
                    raw_vid_ndarr = raw_tif.data()
                    success_load = True
        except:
            print("Failed with ScanImage, trying other methods instead for reading tiff file")
            success_load = False

    if not success_load: # If the ScanImageTiffReader is not working, try using the Tifffile instead
        try:
            raw_tif = TiffFile(filepath)
            raw_vid_ndarr = raw_tif.asarray()
            if (len(raw_tif.shape()) < 3):
                    print("Target file seems to be just a picture, please verify the path given")
                    success_load = False
                    return
        except:
            print("Failed to load the specified tiff file, please check the path and tiff file")
    return raw_vid_ndarr.astype(np.float16)

def load_tif_video(filepath, rotation_k=0, t_last=False):
    ''' Load image from tif file
        Assuming the dims to be HxWxT
    
    Inputs:
    -------
        **filepath** [path]: path leading to the .tiff or .tif inference file
        **rotation_k** [int(0-3)]: clockwise rotation degree (x90) to the inference

    Outputs:
    --------
        **raw_vid** ndarray: raw video in ndarray
    '''
    vidfile = sitk.ReadImage(filepath)
    raw_vid = sitk.GetArrayFromImage(vidfile)
    if(t_last):  # Inputs of the FACT_Net were transposed to be [t,h,w], therefore transpose back if t_last==True
        raw_vid=np.transpose(raw_vid, [1,2,0])
    if (rotation_k != 0):
        np.rot90(raw_vid, rotation_k, [0,1])

    # Note: two lines below are for the matrix storing inferrence results, which have nothing to do with the raw video data
    if (raw_vid.max()>32767): # when a certain dtype problem occurs in the stored data
        raw_vid -= 32768
    
    # infer_shape = infer.shape
    return raw_vid

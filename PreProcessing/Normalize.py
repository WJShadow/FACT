import numpy as np

def normal(input_data):
    '''
    Perform normalization to the input multi-dimensional matrix 

    Inputs:
    -------
        **input_data** TxHxW ndarray: 
    
    '''
    # input_data = input_data.astype(np.float64)
    input_data = (input_data-input_data.min())/(input_data.max()-input_data.min())

    return input_data
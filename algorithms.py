############################################
# Train-test sampling generator wrapper
# Algorithms shown in the follwing papers
# 1. VLSW: DOI: 10.1109/JIOT.2019.2909038
# 2. GLW: for comparison purpose
# 3. 'min_input_len' for more train samples
# by Wang Xiangyu
############################################

from sklearn.preprocessing import MinMaxScaler
from algorithm_kernels import *

# Variable Length Sliding Window algorithm
def VLSW(x, name):
    '''
    Initialize default parameters
    (train and test samples share the same paras)
    # Input:
        x - DataFrame: input X with multiple features (timestep,dims)
        name- str: mark output Y one feature of X (timesetp,)
    # Output:
        X - 3D array
        Y - 2D array
        all_case_len - len at each batch of X
        all_case_before_len - len at eacb batch of before_segment 
    '''
    # Pre-set and initialize params for VLSW
    VLSM_params = {
        'dim_in': 10,
        'output_len': 10,           # middle length
        'min_input_len': 8,         # change here to increase train samples
        'max_len': 10
    }
    
    # Initialize total values
    total_x = []
    total_y = []
    total_x_len = []
    total_x_before_len = []
    
    # Initial VLSW parameter
    min_input_len = VLSM_params['min_input_len']
    max_len = VLSM_params['max_len']
    middle_len = VLSM_params['output_len']
    
    x = x.iloc[:,:10]
    y_col_idx = x.columns.get_loc(name)
    # Normalization: compress into (0,1) and df->array
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(np.array(x.values).astype('float32'))
    y = x[:, y_col_idx].reshape(-1,1)

    # Variable length configuration (VL-config: (max_len-min_len)^2 possibilities)
    for input_len in range(min_input_len, max_len + 1):
        case_x, case_y = VLSW_gen_sample(x, y, VLSM_params, input_len, middle_len)
        # shape: (batch,filled_value)
        x_len = np.full(case_x.shape[0], case_x.shape[1])
        x_seq_before_len = np.full(case_x.shape[0], input_len)
        # A tuple of (n_before, n_after) for each dimension
        npad_x = ((0, 0), (max_len - input_len, max_len - input_len), (0, 0))
        sampe_len_x = np.pad(case_x, pad_width=npad_x, mode='constant', constant_values=-10)
        total_x.append(sampe_len_x)
        total_y.append(case_y)
        total_x_len.append(x_len)
        total_x_before_len.append(x_seq_before_len)
    
    # Total x, y
    X = np.concatenate(total_x, axis=0)
    Y = np.concatenate(total_y, axis=0)
    all_case_len = np.concatenate(total_x_len).ravel()
    all_case_before_len = np.concatenate(total_x_before_len).ravel()
    
    # Check NaN
    print('NaN number count: ', np.count_nonzero(np.isnan(X)))
    
    return X, Y, all_case_len, all_case_before_len

# Generic Sliding Window algorithm (truncate)
def GSW(dr, input_len=200, output_len=10):
    
    in_, out_ = [], []
    x = np.array(dr.values).astype('float32')
    
    # Normalization: compress into (0,1) and df->array
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_flat = scaler.fit_transform(np.array(x.values).astype('float32')).flatten()
    x_norm = x_flat.reshape(-1,10)
    
    for i in range(len(x_norm)-input_len-output_len+1):
        in_.append(x_norm[i:(i+input_len),range(x_norm.shape[1])].tolist())
        out_.append(x_norm[(i+input_len):(i+input_len+output_len),range(1)].tolist())
    
    return np.array(in_), np.array(out_)


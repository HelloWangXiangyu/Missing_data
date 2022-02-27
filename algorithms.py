###################################################
# Train-test sampling generator wrapper
# Algorithms shown in the follwing papers
# 1. VLSW: DOI: 10.1109/JIOT.2019.2909038
#    ('min_input_len' for more train samples)
# 2. GLW: for comparison purpose
# 3. Transformer: https://arxiv.org/abs/2103.01600
# by Wang Xiangyu
###################################################
import copy
from tokenize import maybe
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from algorithm_kernels import *
from data_loader import Transformer_Dataset_AM, Transformer_Dataset_SMRT, Transformer_Dataset_index

#------------------------------------ SSIM -------------------------------------#
# Variable Length Sliding Window algorithm
def VLSW(x, name):
    '''Initialize default parameters
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

#------------------------------------ DeepMVI -------------------------------------#
# Test missing block is blackout or not. well-constraint
# This func has some problem to be solved, not very stricted / general.
def is_blackout(matrix):
    arr = (np.sum(np.isnan(matrix).astype(np.int), axis=1) == matrix.shape[1])
    return arr.astype(np.int).sum() > 0

# Generate validation data
def transformer_val(matrix, missing_num):
    ''' "make_validation" in the paper.
    # Input:
        matrix - 2D numpy array: normailized data with NaN
        missing_num - int
    # Output:
    
    '''
    # Padded - following snippet accommodate more complicated situation
    nan_mask = np.isnan(matrix)
    sandw_mat = np.concatenate([np.zeros((1, nan_mask.shape[1])), nan_mask, 
                                np.zeros((1, nan_mask.shape[1]))], axis=0)
    indi_mat = (sandw_mat[1:,:]-sandw_mat[:-1,:]).T
    pos_start = np.where(indi_mat==1)
    pos_end = np.where(indi_mat==-1)
    lens = (pos_end[1]-pos_start[1])[:,None]
    start_index = pos_start[1][:,None]
    time_series = pos_start[0][:,None]
    test_points = np.concatenate([start_index, time_series, lens], axis=1)
    temp = np.copy(test_points[:,2])
    # define block size: ??? why divided by 10
    if (temp.shape[0]>1):
        block_size = temp[int(temp.shape[0]/10):-int(temp.shape[0]/10)-1].mean()
    else:
        block_size = temp.mean()
    w = int(10*np.log10(block_size))
    val_block_size = int(min(block_size,w))
    # ??? missing_num ??? why use val_block_size to divide
    missing_num = int(missing_num/val_block_size)
    train_mat = copy.deepcopy(matrix)
    val_points = []
    
    for _ in range(missing_num):
        # Draw samples from a uniform distribution (low, high, size:block_num)
        validation_points = np.random.uniform(0, matrix.shape[0]-val_block_size, 
                                              matrix.shape[1]).astype(np.int)
        for i,x in enumerate(validation_points):
            train_mat[x:x+val_block_size,i] = np.nan
            val_points.append([x,i,val_block_size])
    return train_mat, matrix, np.array(val_points), test_points, int(block_size), w

# Transformer recovery
def transformer_recovery(input_feats, missing_num):
    '''Main func of temporal transformer method.
    # Input:
        input_feats - 2D numpy array: 10 dim features
                      padded with NaN
    # Output:
        output_feats - 2D numpy array: imputed matrix
    '''
    print('---------- start ----------')
    
    # Normalization
    mean = np.nanmean(input_feats, axis=0)
    std = np.nanstd(input_feats, axis=0)
    input_feats = (input_feats-mean)/std
    
    # Validation data generation
    # ??? 
    missing_num = 10*min(max(int(input_feats.shape[0]/100), 1), 500)
    train_feats, val_feats, val_points, test_points, block_size, kernel_size  = transformer_val(
        input_feats, missing_num)
    
    time_context = min(int(input_feats.shape[0]/2), 30*kernel_size)
    
    use_embed = (not is_blackout(input_feats))
    use_context = (block_size <= kernel_size)
    use_local = (block_size < kernel_size)
    
    print('Block size is %d, kernel size is %d'%(block_size, kernel_size))
    print('Use Kernel Regression : ', use_embed)
    print('Use Context in Keys :', use_context)
    print('Use Local Attention :', use_local)
    
    buffer_size = input_feats.shape[0]*input_feats.shape[1]
    batch_size = min(input_feats.shape[1]*int(input_feats.shape[0]/time_context), 16)
    interval = 1000
    
    # Instantiate data sequence for each split
    train_set_SMRT = Transformer_Dataset_SMRT(train_feats, use_local, time_context=time_context)
    train_set_AM = Transformer_Dataset_AM(train_feats, use_local, time_context=time_context)
    train_set_index = Transformer_Dataset_index(train_feats)
    train_loader_SMRT = tf.data.Dataset.from_generator(
        train_set_SMRT,
        output_types=(
            tf.float32,
            tf.bool,
            tf.float32,
            tf.int32
            ),
        output_shapes=(
            tf.TensorShape([None,]),
            tf.TensorShape([None,]),
            tf.TensorShape([None,None]),
            tf.TensorShape([None])
            )
        ) 
    train_loader_index = tf.data.Dataset.from_generator(
        train_set_index,
        output_types=(tf.int32,
                      tf.int8
                      ),
        output_shapes=(tf.TensorShape([None]),
                       tf.TensorShape(())
                       )
        )
    train_loader_AM = tf.data.Dataset.from_generator(
        train_set_AM,
        output_types=tf.float32,
        output_shapes=tf.TensorShape([None,None])
        )
    
    train_loader_SMRT = train_loader_SMRT.padded_batch(batch_size)
    train_loader_index = train_loader_index.batch(batch_size)
    train_loader_AM = train_loader_AM.padded_batch(batch_size, padding_values=float('-inf'))
    train_loader = tf.data.Dataset.zip((train_loader_SMRT, train_loader_index, train_loader_AM)).shuffle(buffer_size)
    
    print(next(iter(train_loader)))
    '''
    for attn_mask in train_loader_AM:
        print(tf.transpose(attn_mask, perm=(0,2,1)))
    
    for idx, stat_time in train_loader_index:
        print(idx)    
    
    for series, mask, residuals, time_vector in train_loader_SMRT:
        print(list(series))
    
    return 
    '''


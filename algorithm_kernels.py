############################################
# Train-test sampling generator core tools
# Wrapped by algorithms.py
# by Wang Xiangyu
############################################

import numpy as np
import numpy.matlib as mat

# Sample assembly and generate under single length-config for VLSM wrapper
def VLSW_gen_sample(x, y, params, seq_input_len, seq_output_len):
    '''
    Sampling_len = seq_before_len + seq_after_len
    # Input:
        x - array: input X (timestep,dim)
        y - array: output Y - target feature to impute (timestep,)
        params - struc: VLSM parameters
        seq_before_len, seq_after_len, seq_output_len - int: VL-config
        max_length - int: fixed mirrored: (before = after) in max lenth
    # Output:
        input_seq - 3D array: train X under single VL-config
        outout_seq - 2D array: train Y under single VL-config
    '''
    # Shape of samples
    total_samples = x.shape[0]
    sing_sample_len = seq_input_len + seq_output_len + seq_input_len
    
    # input_batch_idx - list: batch len = total_samples - total_len; 
    #                         sample X len = sing_sample_len
    # output_batch_idx - list: sample Y len = seq_output_len
    input_batch_idx = [
        list(
            range(i, i + seq_input_len)
            ) + 
        list(
             range(i + seq_output_len + seq_input_len, i + seq_input_len + seq_output_len + seq_input_len)   
            ) for i in range(total_samples - sing_sample_len + 1)
    ]
    output_batch_idx = [
        list(
            range(j +seq_input_len, j + seq_output_len + seq_input_len)
        ) for j in range(total_samples - sing_sample_len + 1)
    ]
    # Form a 3D array X (batch, sampling_len, dim)
    input_seq = np.take(x, input_batch_idx, axis=0)

    # Pad target features with zeros as seq_middle
    # middle_z = np.zeros((seq_output_len, params['output_len']))
    middle_z = mat.repmat(np.array(-10), seq_output_len, params['output_len'])
    # Embed middle_z: input_seq (batch,sampling_len,dim) -> (batch,sampling_len + output_len,dim)
    input_seq = np.array(
        [np.concatenate((i[:seq_input_len], middle_z, i[seq_input_len:])) for i in input_seq]
        )
    # Form a 2D array Y (batch, sampling_len) 
    output_seq = np.take(y, output_batch_idx, axis=0)

    return input_seq, output_seq
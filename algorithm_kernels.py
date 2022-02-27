############################################
# Train-test sampling generator core tools
# Wrapped by algorithms.py
# by Wang Xiangyu
############################################

import numpy as np
import numpy.matlib as mat
import copy

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

# Data sequence iterator for each split (torch.utils.data.Dataset)
class Transformer_Dataset_SMRT:
    # Generate iterable type dataset for tf.data.Dataset.from_generator
    def __init__(self, feats, use_local, time_context=None) -> None:
        # Initializations
        self.feats = feats
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.size = int(self.feats.shape[0]*self.feats.shape[1])
        self.use_local = use_local
    
    def __len__(self):
        # Donates the total element number
        return self.size
    
    def __getitem__(self, index):
        '''Generate one-element dataset of batch
            # Output: series, mask, residuals, time_vector
        '''
        time_ = (index%self.feats.shape[0])
        tsNumber = int(index/self.feats.shape[0])
        lower_limit = min(time_, self.time_context)
        series = self.feats[time_-lower_limit:time_+self.time_context]
        residuals = np.nan_to_num(copy.deepcopy(self.feats[time_-lower_limit:time_+self.time_context,:]))
        residuals[:,tsNumber] = 0
        time_vector = np.add(range(series.shape[0]), (time_-lower_limit))
        
        series = series[:,tsNumber]
        series = copy.deepcopy(series)
        mask = np.ones(series.shape)
        mask[np.isnan(series)] = 0
        
        series = np.nan_to_num(series)

        context = [tsNumber]
        sz = mask.shape[0]       
        attn_mask = np.ones([450,sz])
        if (not self.use_local):
            attn_mask[:,mask==0] = 0

        attn_mask[attn_mask==0] = float('-inf')
        attn_mask[attn_mask==1] = float(0.0)
        
        return  series, mask>0, residuals, time_vector
        # Context, 0, np.transpose(attn_mask)
    
    def __call__(self):
        for i in range(self.size):
            yield self.__getitem__(i)

# Data sequence iterator for each split (torch.utils.data.Dataset)
class Transformer_Dataset_index:
    # Generate iterable type dataset: index for tf.data.Dataset.from_generator
    def __init__(self, feats) -> None:
        # Initializations
        self.feats = feats
        self.num_dim = len(self.feats.shape)
        self.size = int(self.feats.shape[0]*self.feats.shape[1])
    
    def __len__(self):
        # Donates the total element number
        return self.size
    
    def __getitem__(self, index):
        '''Generate one-element dataset of batch
            # Output: index
        '''
        tsNumber = int(index/self.feats.shape[0])
        idx = [tsNumber]
        
        return idx, 0
    
    def __call__(self):
        for i in range(self.size):
            yield self.__getitem__(i)

# Data sequence iterator for each split (torch.utils.data.Dataset)
class Transformer_Dataset_AM:
    # Generate iterable type dataset: attention mast for tf.data.Dataset.from_generator
    def __init__(self, feats, use_local, time_context=None) -> None:
        # Initializations
        self.feats = feats
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.size = int(self.feats.shape[0]*self.feats.shape[1])
        self.use_local = use_local
    
    def __len__(self):
        # Donates the total element number
        return self.size
    
    def __getitem__(self, index):
        '''Generate one-element dataset of batch
            # Output: attention_mask
        '''
        time_ = (index%self.feats.shape[0])
        tsNumber = int(index/self.feats.shape[0])
        lower_limit = min(time_, self.time_context)
        series = self.feats[time_-lower_limit:time_+self.time_context]
        
        series = series[:,tsNumber]
        series = copy.deepcopy(series)
        mask = np.ones(series.shape)
        mask[np.isnan(series)] = 0
        
        sz = mask.shape[0]       
        attn_mask = np.ones([450,sz])
        if (not self.use_local):
            attn_mask[:,mask==0] = 0

        attn_mask[attn_mask==0] = float('-inf')
        attn_mask[attn_mask==1] = float(0.0)
        
        return np.transpose(attn_mask)
    
    def __call__(self):
        for i in range(self.size):
            yield self.__getitem__(i)
            
''' give up the following iterator due to Keras does not provide iterable type dataset sampler
class Transformer_Dataset(keras.utils.Sequence):
    # Generates data for Keras model train
    def __init__(self, feats, use_local, time_context=None, shuffle=True, batch_size=16):
        # Initializations
        self.batch_size = batch_size
        self.feats = feats.astype(np.float)
        self.num_dim = len(self.feats.shape)
        self.time_context = time_context
        self.size = int(self.feats.shape[0]*self.feats.shape[1])
        self.use_local = use_local
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        # Donates the number of batches of the batch: samples/batch_size
        return self.size
    
    def __getitem__(self, index):
        #Genrates one batch of data
            # Output: series, mask, index, residuals, start_time, time_vector, attn_mask
        index = self.indices[index]
        time_ = (index%self.feats.shape[0])
        tsNumber = int(index/self.feats.shape[0])
        lower_limit = min(time_,self.time_context)
        series = self.feats[time_-lower_limit:time_+self.time_context]
        residuals = np.nan_to_num(copy.deepcopy(self.feats[time_-lower_limit:time_+self.time_context,:]))
        residuals[:,tsNumber] = 0
        time_vector = tf.range(series.shape[0])+(time_-lower_limit)
            
        series = series[:,tsNumber]
        series = copy.deepcopy(series)
        mask = np.ones(series.shape)
        mask[np.isnan(series)] = 0
        
        series = np.nan_to_num(series)

        context = [tsNumber]
        sz = mask.shape[0]
        attn_mask = np.ones([450,sz])
        if (not self.use_local):
            attn_mask[:,mask==0] = 0
            attn_mask = tf.convert_to_tensor(attn_mask)
        #attn_mask = 
        attn_mask = tf.where(
            tf.equal(
                tf.where(
                    tf.equal(attn_mask,0),float('-inf'),attn_mask),1),float(0.0), attn_mask)
        
        return tf.convert_to_tensor(series),\
               tf.convert_to_tensor(mask>0),\
               context,\
               tf.convert_to_tensor(residuals),\
               0,\
               time_vector,\
               tf.transpose(attn_mask)
    
    def on_epoch_end(self):
        # Updates indices after each epoch
        self.indices = np.arange(self.size)
        if self.shuffle == True:
            np.random.shuffle(self.indices)
''' 


           
#################################################
# Prepare train-test datasets from raw DataFrame
# Some data visualization functions for plotting
# by Wang Xiangyu
#################################################
from contextvars import Context
import copy
import pandas_datareader.data as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy.matlib as mat
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
############################### DATA PREPARISON ###############################
#----------------------------------- SSIM ------------------------------------#
# Collect raw data "Adj Close" into feature table
def prep_dr(pick_ten):

    # Define five-year start-end dates: fixed internal
    start_date = '2017-01-06'
    end_date = '2021-12-16'
    
    # Collect ten 'Adj CLose' cols into adj as raw data aquisition; Return: adj
    adj = pd.concat(
        [pd.DataFrame(
            web.DataReader(
                name=pick_ten[i], 
                data_source='yahoo',
                start=start_date, 
                end=end_date)['Adj Close']) for i in range(len(pick_ten))
         ],
        axis=1,
        ignore_index=True
    )
    
    # Calculate daily returns in terms of 'Adj Close'; Return: dr
    dr = adj.pct_change()
    dr.columns = pick_ten
    
    return dr.dropna()

# Missing data generater
def gen_md(df, missing_rate=0.05, length=10):
    '''
    # Input: 
        df - DataFrame: daily return table
        missing rate - float: missing 60 days
        len - integer: before/after input length
    # Output:
        df_train - DataFrame: train_set (dates-60d,10)
        test_seq_X_sc - array: (6,30,10) scaled before/after segments
                          random inside 6 segmets
                          len: before-dummy-after=30
                          dim=10
        test_seq_Y_sc - missing data: scaled value
        test_seq_Y - missing data: real value (6,10,10)
        df - original df padded with nan(missing data)
        test_Y_idx - missing rows index
    '''
    missing_dates = np.round(int(len(df)*missing_rate), -1)
    seg = np.round(int(missing_dates/length))
    seg_len = np.round(int(len(df)/seg))
    # Random local starting point
    rand_lsp = np.random.randint(seg_len-3*length, size=seg)
    test_X_idx = [
        list(
            range(j-10,j))+ 
        list(
            range(j+length,j+2*length)
        ) for j in rand_lsp+range(length, len(df), seg_len)]
    test_Y_idx = [list(range(i, i+length)) for i in rand_lsp+range(length, len(df), seg_len)[:seg]]    
    
    scaler = MinMaxScaler(feature_range=(0,1))
    test_x_sc = scaler.fit_transform(np.array(df.values).astype('float32'))
    test_seq_x_sc = np.take(test_x_sc, test_X_idx, axis=0)
        
    test_seq_X_sc = np.array(
        [np.concatenate(
            (k[:length], mat.repmat(np.array(-10), length, df.shape[1]), k[length:])
            ) for k in test_seq_x_sc]
        )
    test_seq_Y_sc = np.take(test_x_sc, test_Y_idx, axis=0)
    test_seq_Y = np.take(df.values, test_Y_idx, axis=0)
    df_train = df.drop(df.index[np.array(test_Y_idx).reshape(1,-1).ravel()])
    df.iloc[np.array(test_Y_idx).reshape(1,-1).ravel(),:] = np.nan
    
    return df_train, test_seq_X_sc, test_seq_Y_sc, test_seq_Y, df, test_Y_idx

# Data splitting: df -> train-test sets
def process_df(df, split_ratio=1):
    '''
    # Input: 
        df - DataFrame: dataset table
        stock - string: stock id
    # Output:
        df_train - DataFrame: train_set
        dr_test - DataFrame: test_set

    '''
    ## Split df into df_train and df_test with a split ratio
    df_len = len(df.index)
    # Point for splitting df into df_train and df_test
    splt = int(df_len*split_ratio)
    df_train = df.iloc[range(splt),:]
    df_test = df.iloc[splt:,:]
    # Checkpoint
    print('df:{}'.format(df.shape))
    print('df_train:{}'.format(df_train.shape))
    print('df_test:{}'.format(df_test.shape))
        
    return df_train, df_test

#---------------------------------- DeepMVI ----------------------------------#
# MCAR / Blockout missing data generator; similar format to "gen_md"
def gen_md_transformer(df, missing_rate=0.05, length=10):
    '''
    Missing blocks were padded with NaN
    # Input:
        df - DataFrame: daily return
    # Output:
        matrix_pad_nan - 2D  numpy array with NaN
        missing_num - int :60
    '''
    missing_rows = np.round(int(len(df)*missing_rate), -1)
    seg_num = int(missing_rows/length)
    seg_len = int(len(df)/seg_num)
    # Randomly pick a start point of corres local seg
    rand_sp = np.random.randint(seg_len-3*length, size=seg_num)
    nan_idx = [list(range(i, i+length)) for i in rand_sp+range(length, len(df), seg_len)[:seg_num]]
    df.iloc[np.array(nan_idx).flatten(), :] = np.nan
    return df.values, seg_num*len(df.columns)   #return matrix_pad_nan

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


           
############################### DATA VISUALIZATION ###############################
# Plot missing data
def md_bar_plot(df, metric):
    # Creat dummy cols for the obervation
    mask1 = df[metric].values > 0
    mask2 = df[metric].values < 0
    plt.figure(figsize=(24,10))
    plt.title(metric)
    plt.ylabel('Daily Retures')
    plt.xlabel('Years')
    is_nan = df[metric].isna()
    n_groups = is_nan.ne(is_nan.shift()).cumsum()
    gap_list = df[is_nan].groupby(n_groups).aggregate(
    lambda x: (
        x.index[0] + pd.DateOffset(days=-1),
        x.index[-1] + pd.DateOffset(days=+1)
        )
    )[metric].values
    plt.bar(df.index[mask1], df[metric][mask1], color='g')
    plt.bar(df.index[mask2], df[metric][mask2], color='r')
    # resuls in list which contains tuples of each gap start and stop datetime
    gaps = gap_list
    for gap in gaps:
        plt.axvspan(gap[0], gap[1], facecolor='k', alpha=.3) 

    plt.show()
    
# Pandas bar plot
def bar_plot(df, metric):
    '''
    # Input:
        df - DataFrame: csv raw data
        metric - String: stock id
    '''
    # Creat dummy cols for the obervation
    df['positive'] = df[metric] > 0
    ax = df[metric].plot(kind='bar',
                       figsize=(24,10),
                       color=df.positive.map({True:'g', False:'r'}),
                       title=metric
                       )
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Return')
        
    # Make the most of the ticklabels empty so that not too crowded
    ticklabels = ['']*len(df.index)
    # Every step th ticklabel shows the month and day
    ticklabels[::40] = [iter.strftime('%b %d') for iter in df.index[::40]]
    # Every step th ticklabel includes the year
    ticklabels[::120] = [iter.strftime('%b %d\n%Y') for iter in df.index[::120]]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(ticklabels))
    # ax.set_xticks(ax.get_xticks()[::10])
    plt.gcf().autofmt_xdate()    
    plt.show()

# Matplotlib bar plot
def train_test_norm_bar_plot(df_x, df_y, metric):
    '''
    # Input:
        x, y - DataFrame: train, test datasets
    '''
    df_x_col = df_x[metric]
    df_y_col = df_y[metric]
    train_x = np.array(df_x_col.values).astype('float32')
    train_y = np.array(df_y_col.values).astype('float32')
    train_x_max = np.abs(train_x).max(axis=0)
    train_y_max = np.abs(train_y).max(axis=0)

    x_norm = np.divide(train_x, train_x_max)
    y_norm = np.divide(train_y, train_y_max)

    plt.figure(figsize=(24,10))
    plt.bar(range(len(x_norm)), x_norm, label='train')
    plt.bar(range(len(x_norm), len(x_norm)+len(y_norm)), y_norm, label='test')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)

    plt.show()
    
def history_plot(history, metric, name):
    train = history.history[metric]
    valid = history.history['val_'+ metric]
    
    plt.plot(train, marker=".", markersize=6, label='Train Loss'), 
    plt.plot(valid, marker=".", markersize=6, label='Validation Loss')
    plt.ylabel('Loss (%s)' %name)
    plt.xlabel('Training Epochs')
    plt.title('Train vs. Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(axis='y')
    plt.show()
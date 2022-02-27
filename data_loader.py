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
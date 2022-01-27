############################################################
# Implement Algorithm SSIM with tensorflow 2
# DOI: 10.1109/JIOT.2019.2909038
# Github repositary: https://github.com/ivivan/SSIM_Seq2Seq.git
# Part I: VLSW sampling
# Part II: SSIM training
# Part III: Imputing and evaluating
# Specify targeting imputed stock id in VLSW sampling
# by Wang Xiangyu
#############################################################

from data_loader import *
from algorithms import GSW, VLSW
from model_eval import md_pred

import numpy as np
import matplotlib.pyplot as plt

import pickle as pkl
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Masking, Bidirectional, LSTM, Dense, \
    RepeatVector, TimeDistributed, Input, concatenate, Activation, dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from keras.callbacks import EarlyStopping
import pydot as pyd
from keras.utils.vis_utils import plot_model, model_to_dot
keras.utils.vis_utils.pydot = pyd

####################################################################
############################### VLSW ###############################
####################################################################

# Pick ten stocks from Top30 HSI components manually
pick_ten = [
    '0012.HK', '0003.HK',
    '0017.HK', '0002.HK',
    '1038.HK', '0883.HK',
    '1299.HK', '0101.HK',
    '1109.HK', '2018.HK'
    ]
# Prepare daily return DataFrame or skip here and direct-read saved data
# OBTAIN ONLINE DATA
dr = prep_dr(pick_ten)

'''# Save dataframe into data folder
dr.to_pickle('./data/dailyreturn.pkl')

# Read just saved daily return
dr = pd.read_pickle('./data/dailyreturn.pkl')
# Plot complete daily return bar chart: remv-hash and run
bar_plot(dr, '0012.HK')       # Original data
'''
# Generate missing 5% data
df_train, x_test_sc, y_test_sc, y_test, df_train_m, y_test_ind = gen_md(dr)

# Plot missing-data daily return bar chart
md_bar_plot(df_train_m, '0012.HK')     # After data missing

# train-test data split with ratio
#df_train, _ = process_df(dr)

# Plot normalized the train-test data
# train_test_norm_bar_plot(df_train, df_test, '0012.HK')

# Apply V-LSW to generate train-test data
## Specify the targeting stock you want to impute
x_train, y_train, x_train_len, x_train_before_len = VLSW(
    df_train, 
    '0012.HK'
    )
''' # Treat missing data as test set
x_test, y_test, x_test_len, x_test_before_len = VLSW(
    df_test, 
    df_test['0012.HK']
    )

'''
# Generic method G-SW to generate training data set
# x_train, y_train = GSW(dr, input_len=100, output_len=10)

''' # for model check & debug
X_ = np.load('./data/X_demo.npz')
x_train = X_['X']
x_train = x_train[:,:,range(x_train.shape[2]-1)]
y_train = X_['Y']
y_train = y_train[:,:,range(x_train.shape[2]-1)]

'''

# Check train set
print('x_train:{}'.format(x_train.shape))
print('y_train:{}'.format(y_train.shape))

####################################################################
############################ SSIM MODEL ############################
####################################################################

####### Hyperparams #######
n_hidden = 50

####### Input Layer #######
input_train = Input(shape=(x_train.shape[1], x_train.shape[2]))
output_train = Input(shape=(y_train.shape[1], y_train.shape[2]))
# Mask layer: pre-set mask value = -10 as dummy
Encoder_input = Masking(mask_value=-10)(input_train)
# print(input_train)
####### Encoder biLSTM #######
# output (none,30,100) of biLSTM: stack hidden state
encoder_stack_h, h_fw, c_fw, h_bw, c_bw = Bidirectional(
    LSTM(
        n_hidden, 
        activation='tanh', 
        dropout=0.1, 
        recurrent_dropout=0,
        return_state=True, 
        return_sequences=True
        ))(Encoder_input)
# Output_layer for Encoder (none,100)
encoder_h = Dense(n_hidden*2)(tf.concat([h_fw, h_bw], axis=1))
encoder_c = Dense(n_hidden*2)(tf.concat([c_fw, c_bw], axis=1))

####### Decoder LSTM #######
# Repeat the hidden state as the input of hidden layer
decoder_input = RepeatVector(output_train.shape[1])(encoder_h)
# Ouput stack hidden state of encoder without return states
decoder_stack_h = LSTM(
    n_hidden*2, 
    activation='tanh', 
    dropout=0.1, 
    recurrent_dropout=0,
    return_state=False,
    return_sequences=True
    )(decoder_input, initial_state=[encoder_h, encoder_c])

####### Attention Layer #######
# Using "Dot" method proposed by Luong's paper
score = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])    
attenW = Activation('softmax')(score)
context = dot([attenW, encoder_stack_h], axes=[2,1])            
decoder_combined_context = concatenate([context, decoder_stack_h])
# Temporal slice of the input
out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)
# Build model
model = Model(inputs=input_train, outputs=out)
# Compile model
opt = Adam(learning_rate=0.001, clipnorm=1)
model.compile(loss='mean_squared_error', 
              optimizer=opt, 
              metrics=['mae'])
model.summary()
#model.save('./data/model_SSIM.h5')
'''# plot model architecture - just illustrate not accurate enough
plot_model(model, to_file='./img/SSIM_model_architec.png', 
           show_shapes=True, show_layer_names=True)
'''
# Train model
es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
history = model.fit(x_train, y_train, validation_split=0.2,
                    epochs=100, callbacks=[es], batch_size=100)

####################################################################
############################ MODEL EVAL ############################
####################################################################

# Training and validation loss - regression check
#history_plot(history, 'loss', 'MSE')
history_plot(history, 'mae', 'MAE')

# Missing data imputation and evaluation
test_pred = md_pred(model, x_test_sc, y_test, dr, '0012.HK')

print(test_pred)
print(y_test[:,:,0].squeeze()) 


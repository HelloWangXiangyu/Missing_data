############################################
# Missing data imputation and evaluation
# Architectures shown in the follwing papers
# 1. SSIM: DOI: 10.1109/JIOT.2019.2909038
# by Wang Xiangyu
############################################

# Missing data imputation with model.predict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Calculate smape
def smape_cal(A, F):
    '''
    # Input:
        A - row vec: actual
        F - row vec: forecasr
    # Output:
        smape - float
    '''
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Metrics evaluation
def eval(y_true, y_pred):
    '''
    # Input: 
        y_true - array: 2D
        y_pred - array: 2D
    # Output:
        metrics
    '''
    # y_true = y_true[:,:,0].squeeze()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rsme = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = smape_cal(y_true, y_pred)
    
    return mse, mae, rsme, mape, smape

# Missing data predict
def md_pred(model, x_test_sc, y_test, dr, name ):
    '''
    # Input: 
        model - tensor
        x_test_sc - array: scaled input test (6,30,10) 
        dr_col - DataFrame: daily return to be imputed
    # Output:
        test_pred - array: predict missing data of one dim (6,10)
    '''
    dr_col = dr[name]
    dist = max(dr_col.values) - min(dr_col.values)
    # Output predicted daily returns (6,10) of one stock
    test_pred_sc = model.predict(x_test_sc).squeeze()
    test_pred = test_pred_sc*dist + min(dr_col.values)
    
    y_test = y_test[:,:,dr.columns.get_loc(name)].squeeze()
    
    for i in range(test_pred.shape[0]):
        plt.figure()
        plt.title('Ground Truth vs. Imputation')
        plt.ylabel('Daily Retures')
        plt.xlabel('Missing data time series')
        plt.bar(range(len(test_pred[i,:])),
                test_pred[i,:], color='orange', 
                alpha=0.5, label='Imputation')
        plt.bar(range(len(y_test[i,:])), 
                y_test[i,:], color='green', 
                alpha=0.3, label='Ground Truth')
        plt.legend(loc='best')

        plt.show()
    # Evaluate
    mse, mae, rsme, mape, smape, = eval(y_test, test_pred)
    
    print('MSE: ', mse)
    print('MAE: ', mae)
    print('RSME: ', rsme)
    print('MAPE: ', mape)
    print('SMAPE: ', smape)
    
    return test_pred

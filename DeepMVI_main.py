############################################################
# Implement Algorithm DeepMVI with tensorflow 2
# https://arxiv.org/abs/2103.01600
# Github repositary: https://github.com/pbansal5/DeepMVI.git
# Part I: Multi-series MCAR daily return to be recovered
# Part II: Matrix recoverd by temporal transfer
# Part III: 
# Specify targeting imputed stock id in VLSW sampling
# by Wang Xiangyu
#############################################################

from data_loader import *
from algorithms import *

####################################################################
########################## Missing Value ###########################
####################################################################
'''
Based on the missing scenario description of this paper, 
Missing Completely at Random (MCAR) or Blockout senario was conducted.
Each incomplete time series has 5% of its data missing. 
The missing data is in randomly chosen blocks of constant 
size of 10.
'''
# Load raw data - dr and export .npy or .txt
dr = pd.read_pickle('./data/dailyreturn.pkl')
#bar_plot(dr, '2018.HK')
#np.savetxt('./data/original_matrix.txt', dr.values)
# Generate missing data padded with NaN (blockout)
matrix, missing_num = gen_md_transformer(dr)
#np.savetxt('./data/dr_matrix.txt', matrix)

####################################################################
########################### Data Recovery ##########################
####################################################################
'''
Core algorithms imported from "algorithm.py". 
Temporal transformer model classes built inside.
'transformer_recovery' acts as 'recover_matrix' in the original paper.
'''
matrix_imputed = transformer_recovery(matrix, missing_num)

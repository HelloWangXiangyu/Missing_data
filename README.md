# Problem 1 - Missing data

![](https://progress-bar.dev/35/?title=Overall) ![](https://img.shields.io/badge/windows-passing-greenn)  
Algorithms implementation for missing data imputation.  


## Environment
```bash
Python 3.9.7
TensorFlow 2.7.0
CUDA 11.2.2
cudnn 8.2.1
```

## Usage
- Data load options
  - Option 1: Remote data access pakage
   ```bash
   pip install pandas-datareader==0.10.0
   ``` 
   
   ```py
   dr = prep_dr(pick_ten)
   ```
  - Option 2: Read local pre-saved DataFrame
   ```py
   import pandas as pd

   dr = pd.read_pickle('./data/dailyreturn.pkl')
   ```

- Run main  
```base
python SSIM_main.py
python DeepMVI_main.py
```

## Progress Report
### (**2021-12-17 --- 2022-12-23**)
> - [x] TA duty of an MSc course MESF 5520  
   (Final exam grading/lab demo)

### (**2021-12-24 --- 2022-01-12**)
> - [x] Background theory/knowledge organizing
> - [x] Papers/books reading on RNNs 
> - [x] Code (original paper) debugging
> - [x] Github repository 

### (**2022-01-13 --- 2022-01-28**)
> [Paper 1: SSIM—A Deep Learning Approach for Recovering Missing Time Series Sensor Data](https://ieeexplore.ieee.org/document/8681112)
> - [x] VLSW train sample generation algorithm
> - [x] Seq2Seq with Luong global attention model
> - [x] Missing data imputation and evaluation
> - [x] Paper chapter drafting  

### (**2022-01-28 --- 2022-02-28**)  
> [Paper 2: Missing Value Imputation on Multidimensional Time Series](https://arxiv.org/abs/2103.01600)
> - [x] Literature/code review on attention mechanism and transformer model
> - [x] Literature/code review on GANs
> - [x] Sample generator coding  

## To-Do List
Paper 2: Missing Value Imputation on Multidimensional Time Series
- [ ] Temporal transformer model (under-going)
- [ ] Paper chapter drafting

Paper 3: Learning from Irregularly-Sampled Time Series: A Missing Data Perspective
- [ ] Partial encoder-decoder GAN model
- [ ] Paper chapter drafting

Paper 4: NAOMI: Non-autoregressive Multiresolution Sequence Imputation
- [ ] Non-autogressive GAN model
- [ ] Paper done

Problem 2) SDE GAN

## Comment
```bash  
>> My first deep learning project:
Challenging but interesting!
Really enjoy the journey!
```

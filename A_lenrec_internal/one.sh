#!/bin/bash

#1.Choose conda environment for simulation
conda activate lens

#2.Run CMB and noise simulations
python -W ignore sims.py -np 4    #-W ignore 忽略脚本中的警告信息   #-np 30 表示脚本使用 30 个进程来执行其任务

#3.replace the input map(i.e. the observed data map, for example: data_TQU_map.fits) with the last simulated map(for example: map_TQU_1024_0012.fits)
mv data_TQU_map.fits map_TQU_1024_0012.fits

#4.Choose conda environment for reconstruction
conda activate alilens

#5.Run reconstruction process, get reconstruction potential power spectrum, SNR, and covariance matrix
python plot.py

#6.Calculate the alm of QE and mean-field, and reconstruction noise power spectrum (N0)
python products.py

#7.Generate reconstruction potential map based on the klm, and plot
python klmplt.py

##################################################################################################
##########################################    NOTICE    ##########################################
#0.sims.py要用lens环境(因为需要lenspyx-2.0.0版本及以上)，剩下的用alilens环境(因为需要lenspyx-1.0.0版本)
#1.修改 library_parameter.py 中的模拟数以及地址, one.py中的 reconstruction config
#2.修改 klmplt.py中的 input deflection map 的地址， 应为模拟map的最后一张(对应真实观测天图)
#3.修改 products.py 中 params.py 和 map 的地址
#4.sims.py耗时较长，且为多进程，建议提交作业以加快速度。plot.py为主要计算程序，耗时最长(18小时/nsim=300)，且为单进程，必须提交作业以防killed！！！products.py和klmplt.py耗时较短，且为单进程可在本地运行。


###################################################################################################
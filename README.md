# We-love-lensing

![image](https://github.com/ChenWenzheng99/We-love-lensing/blob/main/image/deflection.png)

## This package include the following four parts:

Ⅰ. A full pipeline of the lensing reconstruction, including the simulation of CMB, lensing potential and noise maps. Internal reconstruction with CMB and external reconstruction with LSS tracers (e.g. CIB, galaxy number density).

Ⅱ. Foreground map simulation based on a seven-parameter model.

Ⅲ. CMB B-mode delensing, with two delensing methods (Gradient-order template method and Inverse-lensing method). Include not only the baseline simulation, but also signal+noise simulation used for debiasing. Auto- and cross- power spectra are calculated.

Ⅳ. Parameter constraint. This mainly relys on Cobaya.


## Dependencies:
Ⅰ. Lensing reconstruction:
1. Plancklens
2. lenspyx-1.0.0(for Env.(alilens))
3. lenspyx-2.0.0(for Env.(lens))
4. Healpy
5. Numpy
6. Pylab
7. Mpi4py(Optional)
   
Ⅱ.Foreground simulation:
1. Astropy
2. PySM3
3. pymaster
   
Ⅲ. CMB B-mode delensing
1. CMBlensplus
   
Ⅳ. Parameter constraint
1. Cobaya
2. CAMB


## The structure of Lensing reconstruction is as follows:

1. Reconstruction_2048_Simons
   We simulate the CMB maps, Phi maps and the instrumental noise at 145 GHz.
   You can perform reconstruction with this folder, but it maybe extrmely slow if there were too many simulations.
   Because we handle the simulation in only 1 groups.
   
2. Reconstruction_2048_Simons2
   We simulate the instrumental noise at 93 GHz.
   You can perform reconstruction with this folder, but it maybe extrmely slow if there were too many simulations.
   Because we handle the simulation in only 1 groups.

3. Reconstruction_multi_process
   We don't perform simulation at 93 GHz in this folder, but it could have been able to. So prepare the simulations ahead.
   The 500 sets of simulations are divided into 5 groups, we run the five groups simultaneously to get the raw reconstructed alm, and then move them to /process_added/ALILENS/temp/qlm_dd,
   where we then calculate the MF and N0, post-process(subtract the MF and normalization) and store the 500 reconstructed qlms.

4. Reconstruction_multi_process2
   We don't perform simulation at 145 GHz in this folder, but it could have been able to. So prepare the simulations ahead.
   The 500 sets of simulations are divided into 5 groups, we run the five groups simultaneously to get the raw reconstructed alm, and then move them to /process_added/ALILENS/temp/qlm_dd,
   where we then calculate the MF and N0, post-process(subtract the MF and normalization) and store the 500 reconstructed qlms.
   Besides, there are also external reconstruction files in /process_added, we copy them to /External reconstruction.

5. External reconstruction
  This folder contains the simulation of LSS tracers, such as CIB, galaxy number density. External reconstruction is based on these LSS tracers.
  The combination between the external reconstruction and the internal reconstruction will improve the SNR of the phi reconstruction, especially on small scales, under current       
  instrumental noise level. This will therefore improve the delensing efficiency.

The code structure of one group internal reconstruction (e.g. the total Reconstruction_2048_Simons, the total Reconstruction_2048_Simons2, or every single process in Reconstruction_multi_process and Reconstruction_multi_process2) is as follows:
![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/simons1.png)

The overall code structure of the 93GHz + 145GHz + CIB + LSST reconstruction is as follows:
![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/multi.png)

The overall pipeline of the CMB delensing is as follows:
![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/pipeline.png)

# :sparkling_heart: :sparkling_heart: :sparkling_heart: Lensflow :sparkling_heart: :sparkling_heart: :sparkling_heart:

![image](https://github.com/ChenWenzheng99/We-love-lensing/blob/main/image/deflection.png)
(Figure from https://www.esa.int/Science_Exploration/Space_Science/Herschel/Herschel_helps_find_elusive_signals_from_the_early_Universe)

## This package include the following four parts :clap: :

Ⅰ. A microwave sky simulation suite, including CMB, foreground and noise.

Ⅱ. A internal linear combination (ILC) suite, performed in several domains, with or without component deprojection.

Ⅲ. A standard template of **the lensing reconstruction**. Internal reconstruction with CMB is performed woth **plancklens** and external reconstruction with LSS tracers (e.g. CIB, galaxy number density fluctuation) is based on 2-pt function.

Ⅳ. A sbatch script of **CMB B-mode delensing** based on simulation, with two delensing methods (Gradient-order template method and Inverse-lensing method). Auto- and cross- power spectra of observed B-modes and lensing B-mode template are calculated.

Ⅴ. A **Parameter constraint** example. This mainly relys on Cobaya.


## Dependencies:
Ⅰ. Lensing reconstruction :raised_hands: : 
- Plancklens  (https://github.com/carronj/plancklens)
- Lenspyx-2.0.0(for Env.(lens))  (https://github.com/carronj/lenspyx/releases/tag/v2.0.0)
- Healpy  (https://github.com/healpy/healpy)
- Numpy
- Pylab
- Mpi4py(Optional)
   
Ⅱ.Foreground simulation :raised_hands: :
- Astropy  (https://github.com/astropy/astropy)
- PySM3  (https://pysm3.readthedocs.io/en/latest/index.html#installation)
- Pymaster  (https://namaster.readthedocs.io/en/latest/source/installation.html)
   
Ⅲ. CMB B-mode delensing :raised_hands: :
- CMBlensplus  (https://github.com/toshiyan/cmblensplus)
- Scipy
   
Ⅳ. Parameter constraint :raised_hands: :
- Cobaya  (https://github.com/CobayaSampler/cobaya)
- CAMB  (https://github.com/cmbant/CAMB)


## The structure of Lensing reconstruction is as follows:

5. External reconstruction
  This folder contains the simulation of LSS tracers, such as CIB, galaxy number density. External reconstruction is based on these LSS tracers.
  The combination between the external reconstruction and the internal reconstruction will improve the SNR of the phi reconstruction, especially on small scales, under current       
  instrumental noise level. This will therefore improve the delensing efficiency.


The overall code structure of the 93GHz + 145GHz + CIB + LSST reconstruction is as follows:
![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/multi.png)

The overall pipeline of the CMB delensing is as follows:
![image](https://github.com/ChenWenzheng99/Lensing-tracers/blob/main/image/pipeline.png)

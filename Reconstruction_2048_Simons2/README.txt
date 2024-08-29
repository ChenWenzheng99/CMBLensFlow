This folder is used for instrumental noise @ 93GHz simulation. 

You can also run internal reconstruction use this folder, but this will be extremely slow when there are too many simulations, I recommend to use 'Reconstruction_multi_process', which can divide the simulations into several groups, and process them simultaneously.


Simulation
1. Env.(lens) : Run sims_ali.py with pool, you should 'mpiexec python -W ignore sims_ali.py -np 20' to submit a job to a single server. This will instrumental noise @ 93GHz simulation maps and will store them at 'test/ALILENS/sims'. 
Notice: the CMB maps have been beam-smoothed, be careful.
Warning: do not use sims.py, this may lead to some simulation corrupted when using a pool due to the old version of lenspyx-1.0.0.


Internal Reconstruction (Not recommended) (Assuming we have 500 sets of simulation)
2. Env.(alilens) : 
(1) Configuration :
Ⅰ.'library_parameter.py' : You should first check the simulation(CMB,noise,fg) dir, and then the number of simulation to be used (seeds). 'bias' and 'var' is the number of simulation used for calculate the mean-field and estimate of Cl. See detailed explanation there.

Ⅱ.'params.py' : Just confirm the eps_min, usually 1e-3 is precise enough. The smaller, the slower. 
Wrrning: If you use a homogenous instrumental noise, 1e-4 or 1e-5 may be required.

Ⅲ.'one.py' : Confirm the nsims, lmax_ivf(the lmax of CMB used for phi reconstruction), lmin_ivf(the lmin of CMB used for phi reconstruction), lmax_qlm(the lmax of the reconstruceted phi). You need to balance the lmax and time, the larger, the slower.

Ⅳ.'ali2020_sims.py' : This is the file that read all you simulation(CMB,noise,fg), check them yourself. Since we use the CMB maps from 'Reconstruction_Simons', which have been beam-smoothed, we need to deconvolve and convolve with new beam.


(1) The first step is to calculate the mean-field, response and N0. 
You need to run function 'ninv' in 'sims_ali.py' at least once, to get the 'ninv_p.fits' and 'ninv_t.fits' which lies in 'ALILENS/sims/ninv'.
After that, just run 'mpiexec python lensingrec_quick.py' (Row25-Row34)with single server, you should check the 'qe_key'(ptt,pp,p for TT, pol only, and MV). In this step, you should set 'bias=500' and 'var=0'. This we generate the reconstructed phi alm and store them in 'ALILENS/temp/qlm_dd'， then mean-field alm is just their average. This will be extremely time-consuming, several weeks are needed.

(2) Then store the reconstructed phi alm to 'rec_qlm'. Just run 'mpiexec python lensingrec_quick.py' (Row36-Row44)with single server.

(3) You can never be too carefully to check the reconstruction on map level.(E.g. see README in 'Reconstruction_multi_process')

(4) If you need the estimate of the Cl, then run 'python plot.py', you should check the 'qe_key'(ptt,pp,p for TT, pol only, and MV).
Warning: You are suggested to set 'bias=200' and 'var=200' to save time, but this will need you to delete all the 'xxx_hash.pk' in the 'ALILENS/temp' before you run again.

(5)Env.(lens)  Once you get the reconstructed phi alm from 145GHz and 93GHz, you can inverse-combine them. Run 'inv-combined_rec.py', this will return the combined phi alm and the combined effective N0.
This folder is used for Reconstruction @ 145GHz. 

This receives all the 500 'rec_qlm' from 5 groups, we just calculate mean-field and reconstruceted phi alm here.

Internal Reconstruction (Recommended) 
2. Env.(alilens) : 
(1) Configuration :
Ⅰ.'library_parameter.py' : You should first check the simulation(CMB,noise,fg) dir, and then the number of simulation to be used (seeds). 'bias' and 'var' is the number of simulation used for calculate the mean-field and estimate of Cl. See detailed explanation there.

Ⅱ.'params.py' : Just confirm the eps_min, usually 1e-3 is precise enough. The smaller, the slower. 
Wrrning: If you use a homogenous instrumental noise, 1e-4 or 1e-5 may be required.

Ⅲ.'one.py' : Confirm the nsims, lmax_ivf(the lmax of CMB used for phi reconstruction), lmin_ivf(the lmin of CMB used for phi reconstruction), lmax_qlm(the lmax of the reconstruceted phi). You need to balance the lmax and time, the larger, the slower.



(1) The first step is to calculate the mean-field, response and N0. Just run 'mpiexec python lensingrec_quick.py' (Row25-Row34)with single server, you should check the 'qe_key'(ptt,pp,p for TT, pol only, and MV). In this step, you should set 'bias=500' and 'var=0'. This will generate the mean-field alm and store it in 'ALILENS/temp/qlm_dd'.

(2) Then store the reconstructed phi alm to 'rec_qlm'. Just run 'mpiexec python lensingrec_quick.py' (Row36-Row44) with single server.

(3) You are never too careful to check the reconstruction. Run 'mpiexec python check_map_plot.py -np 30' to plot all the 500 difference maps.
 If there is any odd pattern, you should consider delete this simulation. 
(You had better confirm that the simulation of CMB and phi maps are normal before reconstruction, usually a corrupted CMB and PHI map simulation are to blame.)



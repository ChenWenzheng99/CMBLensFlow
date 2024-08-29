This folder is used for Reconstruction @ 93GHz. 

This is the third group of the 500 simulation, includes idx: 200-320. (Simulation will be performed in 'Reconstruction_Simons' and 'Reconstruction_Simons2')



Internal Reconstruction (Recommended) (Assuming we have 500 sets of simulation, here we use 200-320.)
2. Env.(alilens) : 
(1) Configuration :
Ⅰ.'library_parameter.py' : You should first check the simulation(CMB,noise,fg) dir, and then the number of simulation to be used (seeds). 'bias' and 'var' is the number of simulation used for calculate the mean-field and estimate of Cl. See detailed explanation there.

Ⅱ.'params.py' : Just confirm the eps_min, usually 1e-3 is precise enough. The smaller, the slower. 
Wrrning: If you use a homogenous instrumental noise, 1e-4 or 1e-5 may be required.

Ⅲ.'one.py' : Confirm the nsims, lmax_ivf(the lmax of CMB used for phi reconstruction), lmin_ivf(the lmin of CMB used for phi reconstruction), lmax_qlm(the lmax of the reconstruceted phi). You need to balance the lmax and time, the larger, the slower.

Ⅳ.'ali2020_sims.py' : This is the file that read all you simulation(CMB,noise,fg), check them yourself.


(1) The first step is to calculate the mean-field, response and N0. 
You need to run function 'ninv' in 'sims_ali.py' at least once, to get the 'ninv_p.fits' and 'ninv_t.fits' which lies in 'ALILENS/sims/ninv', then youcan copy them to every groups.
After that, just run 'mpiexec python lensingrec_quick.py' (Row25-Row34)with single server, you should check the 'qe_key'(ptt,pp,p for TT, pol only, and MV). In this step, you should set 'bias=120' and 'var=0'. This we generate the reconstructed phi alm and store them in 'ALILENS/temp/qlm_dd'， then mean-field alm is just their average. This will be a little time-consuming, several days are needed.

(2) Move the '/ALILENS/temp/qlms_dd/sim_p_0xxx.fits'(xxx=200 to xxx=300) to '/process_added/ALILENS/temp/qlms_dd', run './move.sh'


This folder also contains the simulation of LSS tracers, such as CIB, galaxy number density. External reconstruction is based on these LSS tracers.
The combination between the external reconstruction and the internal reconstruction will improve the SNR of the phi reconstruction under current instrumental noise level.
This will therefore improve the delensing efficiency.

1.The simulation and external reconstruction package contain the following files:
(1). generate_LSS.py : Based on the theoretical auto- and cross- power spectra of these tracers(e.g. CMB,CIB,LSST), we calculate some scale factors A and the noise power, the tracer map is simply the sum of the scaled input convergence map and the noise realization from the calculated power spectrum.

(2). LSS_tracer_comb.py : Multiple tracers are combined with coeficeints, which is derived when we maximize the correlation coefficient between the combined tracer and the true lensing convergence. This combination coefficient is something like a Wiener-filter, so the combined tracer is actually the filtered one.

(3). run_LSS_ones.py : The main file where we use the package above to generate the LSS tracers and combine them. 
NOTICE: You should provide the internal reconstructed convergence and the semi-analytical N0 when combining these tracers if CMB reconstructed convergence is involved.

(4). check_map_plot.py : You are suggested to check the combined tracers on map level, change the dir to the combined tracer klm, and the output will be the difference map.
If there is any abnormal map, you should run the corresponding simulation seperately again or just disgard it.


2. How to run?
(1) In run_LSS_ones.py, use the 'run_LSS_generator' function to generate LSS tracer maps, use the 'run_LSS_combinatory' to combine them.
	
(2) In run_LSS_ones.py, use the 'cal_rho' function to calculate the correlation coefficient between the combined tracer and the true lensing convergence of all the simulations.



Simply './submit_edit.sh' to submit several jobs with Env.(lens).

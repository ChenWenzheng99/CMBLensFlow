This folder contains the parameter fitting pipeline. After run post_cal.py in /CMB Delensing, you will get hundreds sets of auto- and cross- power between the observed B-mode and the lensing B-mode template in /fit_package_temp_comb folder. This follows by parameter fitting pipeline.


### Calculate Cov matrix and mean power spectra
Run cells in cal_cov_ones.ipynb from where it begins to where it ends.

### Get noise power
Run cells in test_vector.ipynb to get the cheat noise power (cell [14]). You can also check the chi square value (cell [27]) to see whether the data are promising

### Run Cobaya
Run /fit_package_temp_comb/fit_main.py to fit the data with lensing B-mode template added, return a plot and a chain.pkl.
Run /fit_package_temp_comb/fit_main_cut.py to fit the data without lensing B-mode template added, return a plot and a chain.pkl.
Run /fit_package_temp_comb/fit_main_all.py to plot the two chains in one plot.
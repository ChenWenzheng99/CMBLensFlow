#!/bin/bash

#1.Choose conda environment for simulation
conda activate lens

#2.Calculate the inverse noise map
python -W get_ninv.py -np 4    #

#3.Run reconstruction to get rec qlm
python -W lensingrec_quick.py
# or 
mpirun -np 5 python lensingrec_quick_mpi.py     
# NOTICE: You need to run python -W lensingrec_quick.py at first to generate some necessary files for mpi version reconstruction, you need to run it agian after mpi version reconstruction to extra and save the qlm.fits file.

#4.Calculate power spectrum estimate
python plot.py
# NOTICE: Be sure set 'var' in config.yaml before running plot.py
# If you change 'var' in config.yaml after step 3, you need to clean the old hash files, by running delete_hash.sh, before running step 4.

##################################################################################################
##########################################    NOTICE    ##########################################

# This is only an example script, read carefully and modify it according to your own case.

###################################################################################################
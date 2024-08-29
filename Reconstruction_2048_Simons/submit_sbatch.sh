#! /bin/bash

#=====================================================
#===== Modify the following options for your job =====
#=====    DON'T remove the #! /bin/bash lines    =====
#=====      DON'T comment #SBATCH lines          =====
#=====        of partition,account and           =====
#=====                qos                        =====
#=====================================================

# Specify the partition name from which resources will be allocated  
#SBATCH --partition=ali

# Specify which expriment group you belong to.
# This is for the accounting, so if you belong to many experiments,
# write the experiment which will pay for your resource consumption
#SBATCH --account=alicpt

# Specify which qos(job queue) the job is submitted to.
#SBATCH --qos=regular


# ====================================
#SBATCH --job-name=chenwenzheng



#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=240GB
# SBATCH --exclude=aliws[021-028],aliws005
# SBATCH --mem-per-cpu=2000
# SBATCH --nodelist=aliws021

#SBATCH --output=/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/log/output-%j.log
#SBATCH --error=/sharefs/alicpt/users/chenwz/reconstruction_2048_simons/log/error-%j.log

#or use relative path(some example are listed)
# mpiexec python -u tod_gen4cc.py
# mpirun -np 7 ./cosmomc test.ini
# python as.py

date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)



# mpiexec python -W ignore /sharefs/alicpt/users/chenwz/reconstruction_2048_simons/sims.py -np 10

# mpiexec python -W ignore /sharefs/alicpt/users/chenwz/reconstruction_2048_simons/sims_ali.py -np 20

# mpiexec python /sharefs/alicpt/users/chenwz/reconstruction_2048_simons/plot.py

mpiexec python /sharefs/alicpt/users/chenwz/reconstruction_2048_simons/lensingrec_quick.py


# python /sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/simulation_remap_output/data_origin/ones_NEW1_test.py

# mpiexec python -u /sharefs/alicpt/users/chenwz/Testarea/temporary_test/test.py
date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)


# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/output*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/out.txt
# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/error*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/err.txt




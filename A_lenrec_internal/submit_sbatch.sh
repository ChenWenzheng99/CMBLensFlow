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
#SBATCH --job-name=QE_rec



#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=220GB
# SBATCH --exclude=aliws[021-028],aliws005
# SBATCH --mem-per-cpu=2000
# SBATCH --nodelist=aliws[021-048]

#SBATCH --output=/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MUST/rec/A_lenrec_internal/log/output-%j.log
#SBATCH --error=/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MUST/rec/A_lenrec_internal/log/error-%j.log

#or use relative path(some example are listed)
# mpiexec python -u tod_gen4cc.py
# mpirun -np 7 ./cosmomc test.ini
# python as.py

date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)





# mpiexec python -W ignore /sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/rec/QE_rec_temperature_SO/sims_ali.py -np 1   ####  need 'lens' environment


python lensingrec_quick.py



# python /sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/revise/rec/QE_HO_temperature_SO/cal_mmDL_sim_Cl.py


# mpiexec python /sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MAIN/rec/QE_rec_temperature_Ali/qlm_comb.py

# python /sharefs/alicpt/users/chenwz/Testarea/PARTIAL_SKY/simulation_remap_output/data_origin/ones_NEW1_test.py

# python plot.py









date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)


# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/output*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/out.txt
# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/error*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/err.txt




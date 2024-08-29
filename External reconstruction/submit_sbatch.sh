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
#SBATCH --nodelist=aliws[030-048]

#SBATCH --output=/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/log/output-%j.log
#SBATCH --error=/sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/log/error-%j.log

#or use relative path(some example are listed)
# mpiexec python -u tod_gen4cc.py
# mpirun -np 7 ./cosmomc test.ini
# python as.py



# python run_LSS_ones.py
# mpiexec -n 10 python run_LSS_tracer_generate.py 340

mpiexec -n 10 python run_LSS_ones.py 490


# mpiexec python /sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/check_map_plot.py -np 10

# mpiexec python /sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/check_sim_map_plot.py -np 30


date +%m-%d_%H-%M
DATE=$(date +%m%d%H%M)

# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/output*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/out.txt
# mv /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/error*.log /afs/ihep.ac.cn/users/w/wangyiming25/tmp/slurmlogs/err.txt


# forecast 为文章用图
# forecast_new 更新了输出
# forecast_new2 更新了生成map
# forecast_new3 弃用了ground1的小尺度mode, 实际没啥作用，因为wiener-filter已经滤掉very noisy modes

# forecast2 仍为原先的map，而输出改为直接计算Cl

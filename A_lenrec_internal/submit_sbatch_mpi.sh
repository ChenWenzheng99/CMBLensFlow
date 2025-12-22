#!/bin/bash
#SBATCH --job-name=plancklens_mpi
#SBATCH --partition=ali
#SBATCH --account=alicpt
#SBATCH --qos=regular
#SBATCH --nodes=5              # 使用10个节点
#SBATCH --ntasks-per-node=1     # 每节点同时运行 5 个任务
#SBATCH --cpus-per-task=30       # 每个MPI任务使用1个CPU核
#SBATCH --mem=240G              # 每节点内存上限，略留余地
# SBATCH --time=48:00:00
#SBATCH --output=plens_%j.out
#SBATCH --error=plens_%j.err
#SBATCH --nodelist=aliws[005-010]

#SBATCH --output=/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MUST/rec/A_lenrec_internal/log/output-%j.log
#SBATCH --error=/sharefs/alicpt/users/chenwz/Testarea/FORECAST/A_NEW_WORK_MUST/rec/A_lenrec_internal/log/error-%j.log



# 激活环境
# source ~/.bashrc
# conda activate alilens

# 启动MPI作业
mpirun -np 5 python lensingrec_quick_mpi.py

# srun -N10 -n100 python lensingrec_quick_mpi.py
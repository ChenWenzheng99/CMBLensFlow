#!/bin/bash

# 循环执行10次
for i in {0..49}; do
    # 编辑 submit_sbatch.sh 文件的第46行，将最后一个参数改为10*i
    sed -i "46s/.*/mpiexec -n 10 python mpi.py $((10*i))/" /sharefs/alicpt/users/chenwz/Testarea/FORECAST/Foreground/Foreground_satellite/test_sbatch/submit_sbatch.sh
    
    # 提交作业
    sbatch submit_sbatch.sh
done

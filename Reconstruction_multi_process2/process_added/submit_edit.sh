#!/bin/bash

# 循环执行40次
for i in {0..49}; do
    # 编辑 submit_sbatch.sh 文件的第46行，将最后一个参数改为10*i
    # sed -i "47s/.*/mpiexec -n 10 python run_LSS_tracer_generate.py $((10*i))/" /sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/submit_sbatch.sh
    sed -i "49s/.*/mpiexec -n 10 python run_LSS_ones.py $((10*i))/" /sharefs/alicpt/users/chenwz/reconstruction_multi_process2/process_added/submit_sbatch.sh

    # 提交作业
    sbatch submit_sbatch.sh
done

#!/bin/bash

# 设置源文件夹和目标文件夹路径
source_folder="/sharefs/alicpt/users/chenwz/reconstruction_multi_process/process3/ALILENS/temp/qlms_dd"
target_folder="/sharefs/alicpt/users/chenwz/reconstruction_multi_process/process_added/ALILENS/temp/qlms_dd"

# 确保目标文件夹存在
mkdir -p "$target_folder"

# 遍历文件并复制和重命名
for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_p_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_p_p_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_p_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_ptt_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_ptt_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

if false; then
for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_x_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_x_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_x_p_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_x_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

for i in $(seq -f "%04g" 0 99); do
    old_name="${source_folder}/sim_xtt_${i}.fits"
    new_index=$(printf "%04d" $((10#$i + 200)))
    new_name="${target_folder}/sim_xtt_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        cp "$old_name" "$new_name"
        echo "Copied and renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done

fi
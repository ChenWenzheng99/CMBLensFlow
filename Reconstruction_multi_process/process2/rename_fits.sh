#!/bin/bash

# 设置文件夹路径
folder_path="/sharefs/alicpt/users/chenwz/reconstruction_multi_process/process2/ALILENS/temp/qlms_dd"

# 遍历文件夹中的文件并重命名

for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_p_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done


for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_p_p_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_p_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done



for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_ptt_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_ptt_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done



for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_x_p_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_x_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done


for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_x_p_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_x_p_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done


for i in $(seq -w 100 199); do
    old_name="${folder_path}/sim_xtt_$(printf "%04d" $i).fits"
    new_index=$(printf "%04d" $((i - 100)))
    new_name="${folder_path}/sim_xtt_${new_index}.fits"
    
    if [ -f "$old_name" ]; then
        mv "$old_name" "$new_name"
        echo "Renamed: $old_name to $new_name"
    else
        echo "File not found: $old_name"
    fi
done
#!/bin/bash

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_train.sh [-g gpu] [-d db_dir] [-a arch_type] [-l loss_type] [-s stage] [-e exp_name]
# Options:
#     -g: GPU device# 
#     -d: Path to the dataset directory
#     -a: Architecture type ("conv" or "rnn")
#     -l: Loss type ("cgan", "wgan", or "lsgan")
#     -s: Stage to start (0 or 1)
#     -e: Experiment name (e.g., "exp1")

# Default values
db_dir="/path/to/dataset/training"
dataset_name="mydataset"
gpu=0
arch_type="conv"
loss_type="wgan"
start_stage=0
exp_name="conv_wgan_exp1"

while getopts "g:d:a:l:s:e:" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              d ) db_dir=$OPTARG;;
              a ) arch_type=$OPTARG;;
              l ) loss_type=$OPTARG;;
              s ) start_stage=$OPTARG;;
              e ) exp_name=$OPTARG;;
       esac
done

feat_dir="./dump/${dataset_name}/feat/train"
dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
normfeat_dir="./dump/${dataset_name}/norm_feat/train"
model_dir="./model/${dataset_name}"
log_dir="./logs/${dataset_name}"

# Stage 0: Feature extraction
if [[ ${start_stage} -le 0 ]]; then
       python extract_features.py --src ${db_dir} --dst ${feat_dir} --conf ${dconf_path}
       python compute_statistics.py --src ${feat_dir} --stat ${stat_path}
       python normalize_features.py --src ${feat_dir} --dst ${normfeat_dir} --stat ${stat_path}
fi

# Stage 1: Model training
if [[ ${start_stage} -le 1 ]]; then
       python train.py -g ${gpu} \
              --data_rootdir ${normfeat_dir} \
              --model_rootdir ${model_dir} \
              --log_dir ${log_dir} \
              --arch_type ${arch_type} \
              --loss_type ${loss_type} \
              --experiment_name ${exp_name} \
              ${cond}
fi
#!/bin/bash
#SBATCH --job-name=nnunet_2d          # 作业名称，可自定义
#SBATCH --partition=gpu               # 分区名称，根据集群实际情况调整
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU
#SBATCH --cpus-per-task=6             # CPU 核数
#SBATCH --mem=24G                     # 主机内存
#SBATCH --time=24:00:00               # 最大运行时间 (24小时，可改)
#SBATCH --output=%x.%j.out            # 标准输出日志文件
#SBATCH --error=%x.%j.err             # 标准错误日志文件

# ========== 训练命令 ==========
source ~/.bashrc
conda activate /project2/ruishanl_1185/zrui6736/envs/py310

export nnUNet_compile=0
export TORCHDYNAMO_DISABLE=1
export nnUNet_n_proc_DA=1
export OMP_NUM_THREADS=4

nnUNetv2_train 001 2d 0 -device cuda

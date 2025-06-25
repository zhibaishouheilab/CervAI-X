#!/bin/bash

# 设置CUDA设备并运行训练任务
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task902_spine 0 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task902_spine 1 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task902_spine 2 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task902_spine 3 --npz
CUDA_VISIBLE_DEVICES=0 nnUNet_train 2d nnUNetTrainerV2 Task902_spine 4 --npz

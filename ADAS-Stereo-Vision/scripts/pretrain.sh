#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 15\
                --batch_size 1\
                --checkpoint pretrain\
                --pre_train\
                --num_workers 1\
                --dataset kitti\
                --dataset_directory sample_data/KITTI_2015\

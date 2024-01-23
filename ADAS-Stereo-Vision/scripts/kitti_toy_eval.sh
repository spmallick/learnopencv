#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --batch_size 1\
                --checkpoint kitti_toy_eval\
                --num_workers 2\
                --eval\
                --dataset kitti_toy\
                --dataset_directory sample_data/KITTI_2015\
                --resume kitti_finetuned_model.pth.tar
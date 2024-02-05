#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0
python main.py  --epochs 400\
                --batch_size 1\
                --checkpoint kitti_ft\
                --num_workers 2\
                --dataset kitti\
                --dataset_directory sample_data/KITTI_2015\
                --ft\
                --resume kitti_finetuned_model.pth.tar
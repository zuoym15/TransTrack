#!/usr/bin/env bash

# python3 -m torch.distributed.launch --nproc_per_node=8 --use_env track_main/main_tracktrainhalf.py  --output_dir ./output --dataset_file cater --coco_path /projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/ --batch_size 2  --with_box_refine --epochs 150 --lr_drop 100
python3 track_main/main_tracktrainhalf.py  --output_dir ./output --dataset_file cater --coco_path /projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/ --batch_size 2  --with_box_refine --resume ./output/cater_v1_024.pth --epochs 100 --lr_drop 66

#!/usr/bin/env bash

python3 track_main/main_trackval.py  --output_dir . --dataset_file cater --coco_path /projects/katefgroup/datasets/cater/raw/aa_s300_c6_m10/ --batch_size 1 --resume output/checkpoint0074.pth --eval --with_box_refine
#!/usr/bin/env bash

python3 track_main/main_trackval.py  --output_dir . --dataset_file mot --coco_path mot --batch_size 1 --resume output/crowdhuman_mothalf.pth --eval --with_box_refine
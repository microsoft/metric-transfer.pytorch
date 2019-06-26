#!/bin/bash

# num_labeled=13000
# pseudo_ratio=0.1

# or 
num_labeled=26000
pseudo_ratio=0.2

# or
# num_labeled=51000
# pseudo_ratio=0.5 
python imagenet-semi.py \
    --arch resnet50 \
    --gpus 1,2,6,7 \
    --num-labeled ${num_labeled} \
    --data-dir /home/liubin/data/imagenet \
    --pretrained checkpoint/pretrain_models/lemniscate_resnet50.pth.tar  \
    --pseudo-dir checkpoint/pseudos_imagenet/instance_imagenet_nc_resnet50/num_labeled_${num_labeled} \
    --pseudo-ratio ${pseudo_ratio} \
    

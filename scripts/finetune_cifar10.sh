#!/bin/bash

num_labeled=250
python cifar-semi.py \
    --gpus 0 \
    --num-labeled ${num_labeled} \
    --pseudo-file checkpoint/pseudos/instance_nc_wrn-28-2/${num_labeled}_T_1.pth.tar \
    --resume checkpoint/pretrain_models/ckpt_instance_cifar10_wrn-28-2_82.12.pth.tar \
    --pseudo-ratio 0.2

#!/bin/bash

set -x 

export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES=5 python unsupervised/cifar.py --lr-scheduler cosine-with-restart --epochs 1270 

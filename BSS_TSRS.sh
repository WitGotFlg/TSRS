#!/bin/bash

# CUB-200-2011
CUDA_VISIBLE_DEVICES=1 python /home/star/data2/ssc/Transfer-Learning-Library-master/examples/task_adaptation/image_classification/bss_tsrs.py -a resnet50_tsrs ./data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log ./logs/bss_tsrs/cub200_100
CUDA_VISIBLE_DEVICES=1 ./bss_tsrs.py -a resnet50_tsrs ./data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log ./logs/bss_tsrs/cub200_50
CUDA_VISIBLE_DEVICES=1 ./bss_tsrs.py -a resnet50_tsrs ./data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log ./logs/bss_tsrs/cub200_30
CUDA_VISIBLE_DEVICES=1 ./bss_tsrs.py -a resnet50_tsrs ./data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log ./logs/bss_tsrs/cub200_15

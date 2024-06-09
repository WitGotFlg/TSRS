#!/bin/bash

# CUB-200-2011
CUDA_VISIBLE_DEVICES=0 python ./bss.py ./data/cub200 -d CUB200 -sr 100 --seed 0 --finetune --log ./logs/bss/cub200_100
CUDA_VISIBLE_DEVICES=0 python ./bss.py ./data/cub200 -d CUB200 -sr 50 --seed 0 --finetune --log ./logs/bss/cub200_50
CUDA_VISIBLE_DEVICES=0 python ./bss.py ./data/cub200 -d CUB200 -sr 30 --seed 0 --finetune --log ./logs/bss/cub200_30
CUDA_VISIBLE_DEVICES=0 python ./bss.py ./data/cub200 -d CUB200 -sr 15 --seed 0 --finetune --log ./logs/bss/cub200_15


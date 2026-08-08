#!/bin/bash
EPOCH_TIME=150
OUT_DIR='./model_out/V2_Final_Fourier_CPEA'
CFG='configs/swin_tiny_patch4_window7_224_lite.yaml'
DATA_DIR='datasets/Synapse'
LIST_DIR='lists/lists_Synapse'
LEARNING_RATE=0.05
IMG_SIZE=224
BATCH_SIZE=24
SEED=42

echo "start train model V2 Final Fourier+CPEA"
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --n_class 9 --list_dir $LIST_DIR --cfg $CFG --root_path $DATA_DIR --max_epochs $EPOCH_TIME --output_dir $OUT_DIR --img_size $IMG_SIZE --base_lr $LEARNING_RATE --batch_size $BATCH_SIZE --seed $SEED

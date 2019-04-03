#!/usr/bin/env zsh

set -x

# Path to configuration file is first argument
source $1

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ${DATAROOT} \
  --name ${NAME} \
  --model ${MODEL} \
  --display_port ${PORT} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --batch_size ${BATCH_SIZE} \
  --num_threads ${NUM_THREADS} \
  --activate_cLR ${ACTIVATE_CLR} \
  --lambda_L1 ${LL1} \
  --lambda_IL1 ${IL1} \
  --lambda_GAN ${LGAN} \
  --lambda_GAN2 ${LGAN2} \
  --lambda_z ${LZ} \
  --lambda_kl ${LKL} \
  --use_dropout \
  --continue-train

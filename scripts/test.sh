#!/usr/bin/env zsh

set -x

# Path to configuration file is first argument
source $1

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ${DATAROOT} \
  --results_dir ${DEST} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --name ${NAME} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --aspect_ratio ${ASPECT_RATIO} \
  --center_crop \
  --no_flip \
  --no_encode

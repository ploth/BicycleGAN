#!/usr/bin/env zsh

set -x

# Path to configuration file is first argument
source $1
TEST_DIR=$2

CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test_folder.py \
  --dataroot ${2} \
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
  --test_dir ${TEST_DIR} \
  --center_crop \
  --no_flip \
  --no_encode

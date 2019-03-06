set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='cars'
NZ=8
DIRECTION='BtoA'
LOAD_SIZE=256
CROP_SIZE=256
SAVE_EPOCH=50
INPUT_NC=1
NITER=200
NITER_DECAY=200

# training
GPU_ID=0
# DISPLAY_ID=$((GPU_ID*10+1))
DISPLAY_ID=-1
PORT=2005
CHECKPOINTS_DIR=./checkpoints/${CLASS}/
DATE=`date --iso-8601=seconds`
NAME=${CLASS}_${MODEL}_${DATE}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
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
  --batch_size 48 \
  --num_threads 12 \
  --use_dropout

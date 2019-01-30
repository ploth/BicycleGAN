set -ex
MODEL='bicycle_gan'
# dataset details
CLASS='packages'
NZ=8
#NO_FLIP='--no_flip'
DIRECTION='BtoA'
LOAD_SIZE=600
CROP_SIZE=600
SAVE_EPOCH=25
INPUT_NC=3
NITER=200
NITER_DECAY=200

# training
GPU_ID=0
DISPLAY_ID=$((GPU_ID*10+1))
PORT=2005
CHECKPOINTS_DIR=../checkpoints/${CLASS}/
DATE=`date --iso-8601=seconds`
NAME=${CLASS}_${MODEL}_${DATE}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
  --display_id ${DISPLAY_ID} \
  --dataroot ./datasets/${CLASS} \
  --name ${NAME} \
  --model ${MODEL} \
  #--display_port ${PORT} \
  --direction ${DIRECTION} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --nz ${NZ} \
  --save_epoch_freq ${SAVE_EPOCH} \
  --input_nc ${INPUT_NC} \
  --niter ${NITER} \
  --niter_decay ${NITER_DECAY} \
  --use_dropout

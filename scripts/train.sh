# Load default configuration
CONFIG=./scripts/config.sh
source $CONFIG

# Make specific configuration
CLASS=$1
DATE=`date --iso-8601=seconds`
NAME=${CLASS}_${MODEL}_${DATE}
DEST=./trainings/$NAME

# Export specific configuration
mkdir -p $DEST
cp $CONFIG $DEST
cat 'CLASS='$CLASS >> $DEST/config.sh
cat 'DATE='$DATE >> $DEST/config.sh
cat 'NAME='$NAME >> $DEST/config.sh

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
  --use_dropout

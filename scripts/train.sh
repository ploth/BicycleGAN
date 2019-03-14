set -x
# Make specific configuration
CLASS=$1
MODEL='bicycle_gan'
DATE=`date --iso-8601=seconds`
NAME=${CLASS}_${MODEL}_${DATE}
DEST=./trainings/$NAME

# Load default configuration
CONFIG=./scripts/config.sh
source $CONFIG

# Export specific configuration
mkdir -p $DEST
TMPFILE=/tmp/config.tmp
echo \
  'MODEL='$MODEL$'\n' \
  'CLASS='$CLASS$'\n' \
  'DATE='$DATE$'\n' \
  'NAME='$NAME$'\n' \
  'DEST='$DEST$'\n' \
  | cat - $CONFIG > $TMPFILE
mv $TMPFILE $DEST/config.sh

# Backup scripts to be safe
cp ./scripts/train.sh $DEST
cp ./scripts/test.sh $DEST

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

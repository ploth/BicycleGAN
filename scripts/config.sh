# General
GPU_ID=0
DISPLAY_ID=-1
PORT=2005
DATAROOT=./datasets/${CLASS}
CHECKPOINTS_DIR=$DEST/checkpoints

# Parameter
DIRECTION='BtoA'
LOAD_SIZE=256
CROP_SIZE=256
NZ=8
SAVE_EPOCH=400
INPUT_NC=3
NITER=200
NITER_DECAY=200
BATCH_SIZE=48
NUM_THREADS=16
NUM_TEST=100
NUM_SAMPLES=10
ASPECT_RATIO=1.0
LL1=10.0 #10
IL1=10.0 #10
LGAN=1.0 #1.0
LGAN2=1.0 #1.0
LZ=0.5 #0.5
LKL=0.01 #0.01

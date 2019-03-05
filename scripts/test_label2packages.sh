set -ex

# dataset
CLASS='packages'
NAME='packages_bicycle_gan_2019-02-01T09:24:07+01:00'
NAME='packages_bicycle_gan_2019-02-04T11:22:29+01:00'
NAME='packages_bicycle_gan_2019-02-18T14:44:49+01:00'
NAME='packages_bicycle_gan_2019-02-19T10:41:09+01:00'
NAME='packages_bicycle_gan_2019-02-26T14:28:41+01:00'
RESULTS_DIR='./results/'$NAME
CHECKPOINTS_DIR='./checkpoints/'${CLASS}'/'
DIRECTION='BtoA'
LOAD_SIZE=256
CROP_SIZE=256
INPUT_NC=3  # number of channels in the input image
ASPECT_RATIO=1.0 # change aspect ratio for the test images

# misc
GPU_ID=0 # gpu id
NUM_TEST=100 # number of input images duirng test
NUM_SAMPLES=10 # number of samples per input images

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} python ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ${CHECKPOINTS_DIR} \
  --name ${NAME} \
  --direction ${DIRECTION} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --aspect_ratio ${ASPECT_RATIO} \
  --center_crop \
  --no_flip \
  --no_encode

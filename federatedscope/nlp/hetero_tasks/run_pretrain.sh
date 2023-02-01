set -e

cd ../../..

DEVICE=$1
DEBUG=False

python federatedscope/main.py \
  --cfg federatedscope/nlp/hetero_tasks/baseline/config_pretrain.yaml \
  outdir exp/atc/pretrain/ \
  device $DEVICE \
  data.is_debug $DEBUG \

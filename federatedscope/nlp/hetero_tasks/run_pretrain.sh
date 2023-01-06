set -e

cd ../../..

DEVICE=$1

python federatedscope/main.py \
  --cfg federatedscope/nlp/hetero_tasks/baseline/config_pretrain.yaml \
  outdir exp/atc/pretrain/ \
  device $DEVICE \

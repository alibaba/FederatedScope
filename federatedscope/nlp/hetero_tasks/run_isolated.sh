set -e

cd ../../..

DEVICE=$1
DEBUG=False

python federatedscope/main.py \
  --cfg federatedscope/nlp/hetero_tasks/baseline/config_isolated.yaml \
  --client_cfg federatedscope/nlp/hetero_tasks/baseline/config_client_isolated.yaml \
  outdir exp/isolated/ \
  device $DEVICE \
  data.is_debug $DEBUG \

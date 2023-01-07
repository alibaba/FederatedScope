set -e

cd ../../..

DEVICE=$1
DEBUG=False

python federatedscope/main.py \
  --cfg federatedscope/nlp/hetero_tasks/baseline/config_fedavg.yaml \
  --client_cfg federatedscope/nlp/hetero_tasks/baseline/config_client_fedavg.yaml \
  outdir exp/fedavg/ \
  device $DEVICE \
  data.is_debug $DEBUG \

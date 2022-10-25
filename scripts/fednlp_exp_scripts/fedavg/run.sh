DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/fedavg/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_fedavg.yaml \
  --client_cfg $CFG_DIR/config_client_fedavg.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \
  data.debug True \

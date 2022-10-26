DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/isolated/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_isolated.yaml \
  --client_cfg $CFG_DIR/config_client_isolated.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \

DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/fednlp/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_fednlp.yaml \
  --client_cfg $CFG_DIR/config_client_fednlp.yaml \
  outdir $EXP_DIR/train/ \
  device $DEVICE \

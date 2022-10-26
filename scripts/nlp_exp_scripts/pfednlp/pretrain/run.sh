DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/pfednlp/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pretrain.yaml \
  outdir $EXP_DIR/pretrain/ \
  device $DEVICE \

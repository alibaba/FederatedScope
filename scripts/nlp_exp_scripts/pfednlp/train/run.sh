DEVICE=$1
CFG_DIR="$( dirname -- "$0"; )"
EXP_DIR="exp/pfednlp/"

python federatedscope/main.py \
  --cfg $CFG_DIR/config_pfednlp.yaml \
  --client_cfg $CFG_DIR/config_client_pfednlp.yaml \
  `#federate.hfl_load_from $EXP_DIR/pretrain/ckpt/` \
  outdir $EXP_DIR/train/ \
  device $DEVICE \

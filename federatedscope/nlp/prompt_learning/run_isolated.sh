set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_isolated.yaml
OUT=exp/isolated/
DEBUG=False

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name cb \
  federate.total_round_num 100 \
  model.prefix_len 120 \
  outdir $OUT/cb \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name copa \
  federate.total_round_num 100 \
  model.prefix_len 20 \
  outdir $OUT/copa \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wsc \
  federate.total_round_num 100 \
  model.prefix_len 60 \
  outdir $OUT/wsc \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name rte \
  federate.total_round_num 100 \
  model.prefix_len 60 \
  outdir $OUT/rte \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wic \
  federate.total_round_num 100 \
  model.prefix_len 20 \
  outdir $OUT/wic \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name multirc \
  federate.total_round_num 50 \
  model.prefix_len 40 \
  outdir $OUT/multirc \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name boolq \
  federate.total_round_num 100 \
  model.prefix_len 40 \
  outdir $OUT/boolq \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name record \
  federate.total_round_num 3 \
  model.prefix_len 40 \
  outdir $OUT/record \
  device $DEVICE \
  data.debug $DEBUG \

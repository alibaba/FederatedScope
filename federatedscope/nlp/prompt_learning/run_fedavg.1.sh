set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_fedavg.yaml
OUT=exp/fedavg/server_train/n_layer_1/
ROUND_NUM=200
ROUND_NUM_RECORD=5
PREFIX_LEN=100
NUM_CLIENT_LAYERS=1
DEBUG=False

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name cb \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/cb \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name copa \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/copa \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wsc \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/wsc \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name rte \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/rte \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wic \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/wic \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name multirc \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/multirc \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name boolq \
  federate.total_round_num $ROUND_NUM \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/boolq \
  device $DEVICE \
  data.debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name record \
  federate.total_round_num $ROUND_NUM_RECORD \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  outdir $OUT/record \
  device $DEVICE \
  data.debug $DEBUG \

set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_fedavg_step.yaml
OUT=exp/fedavg/server_not_train/iid/nc20_nl1_rn100_step/
ROUND_NUM=100
PREFIX_LEN=40
NUM_CLIENT=20
NUM_CLIENT_LAYERS=1
LR=5e-2
MAKE_GLOBAL_TRAIN=False
NON_IID_SPLIT=False
DEBUG=False

LOCAL_STEPS=25
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name boolq \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/boolq \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=10
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name cb \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/cb \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=50
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name copa \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/copa \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=500
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name rte \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/rte \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=100
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wic \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/wic \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=10
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wsc \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/wsc \
  device $DEVICE \
  data.is_debug $DEBUG \

LOCAL_STEPS=150
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name multirc \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/multirc \
  device $DEVICE \
  data.is_debug $DEBUG \

ROUND_NUM=5
LOCAL_STEPS=3250
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name record \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  model.prefix_len $PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/record \
  device $DEVICE \
  data.is_debug $DEBUG \

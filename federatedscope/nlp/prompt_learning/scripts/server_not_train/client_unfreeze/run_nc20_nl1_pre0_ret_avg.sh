set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_client_unfreeze.yaml
OUT=exp/fedavg/server_not_train_2/iid/unfreeze/nc20_nl1_pre0_ret_avg/
ROUND_NUM=100
SERVER_PREFIX_LEN=40
CLIENT_PREFIX_LEN=0
NUM_CLIENT=20
NUM_CLIENT_LAYERS=1
LR=5e-2
MAKE_GLOBAL_TRAIN=False
NON_IID_SPLIT=False
RET_AVG_MODEL=True
DEBUG=False


LOCAL_STEPS=25
python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name boolq \
  data.non_iid_split $NON_IID_SPLIT \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
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
  federate.pl_ret_avg_model $RET_AVG_MODEL \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/multirc \
  device $DEVICE \
  data.is_debug $DEBUG \

#ROUND_NUM=5
#LOCAL_STEPS=3250
#python federatedscope/main.py \
#  --cfg $CFG \
#  data.dataset_name record \
#  data.non_iid_split $NON_IID_SPLIT \
#  federate.client_num $NUM_CLIENT \
#  federate.total_round_num $ROUND_NUM \
#  federate.make_global_train $MAKE_GLOBAL_TRAIN \
#  federate.pl_ret_avg_model $RET_AVG_MODEL \
#  model.server_prefix_len $SERVER_PREFIX_LEN \
#  model.client_prefix_len $CLIENT_PREFIX_LEN \
#  model.num_client_layers $NUM_CLIENT_LAYERS \
#  train.optimizer.lr $LR \
#  train.local_update_steps $LOCAL_STEPS \
#  outdir $OUT/record \
#  device $DEVICE \
#  data.is_debug $DEBUG \

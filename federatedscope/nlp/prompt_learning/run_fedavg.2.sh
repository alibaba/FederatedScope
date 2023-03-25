set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_freeze.yaml
KD_CFG=federatedscope/nlp/prompt_learning/baseline/config_init_kd.yaml
GLOBAL_CFG=federatedscope/nlp/prompt_learning/baseline/config_global.yaml
OUT=exp/server_train_iid_ntpc32/nc20_nl1_ls2/
NUM_TRAIN_PER_CLIENT=32
ROUND_NUM=100
LOCAL_STEPS=2
BATCH_SIZE=16
GRAD_ACCUM=1
SERVER_PREFIX_LEN=40
CLIENT_PREFIX_LEN=40
NUM_CLIENT=20
NUM_CLIENT_LAYERS=1
NUM_CLIENT_LAYERS_PER_CELL=1
LR=5e-2
NON_IID_SPLIT=False
MAKE_GLOBAL_TRAIN=True
SHARE_CLIENT_LAYER_PARAM=False
PL_INIT_KD=False
DEBUG=False


python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name boolq \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/boolq \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name cb \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/cb \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name copa \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/copa \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name rte \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/rte \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wic \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/wic \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wsc \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/wsc \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name multirc \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/multirc \
  device $DEVICE \
  data.is_debug $DEBUG \

python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name record \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.num_train_per_client $NUM_TRAIN_PER_CLIENT \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/record \
  device $DEVICE \
  data.is_debug $DEBUG \
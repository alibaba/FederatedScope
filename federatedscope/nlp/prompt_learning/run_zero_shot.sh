set -e

cd ../../..

DEVICE=$1
CFG=federatedscope/nlp/prompt_learning/baseline/config_zero_shot.yaml
KD_CFG=federatedscope/nlp/prompt_learning/baseline/config_init_kd.yaml
GLOBAL_CFG=federatedscope/nlp/prompt_learning/baseline/config_global.yaml
OUT=exp/zero_shot/opt-1.3b/
ROUND_NUM=1
LOCAL_STEPS=1
BATCH_SIZE=16
GRAD_ACCUM=1
SERVER_PREFIX_LEN=0
CLIENT_PREFIX_LEN=0
NUM_CLIENT=1
NUM_CLIENT_LAYERS=24
NUM_CLIENT_LAYERS_PER_CELL=24
LR=5e-2
NON_IID_SPLIT=False
MAKE_GLOBAL_TRAIN=False
SHARE_CLIENT_LAYER_PARAM=False
PL_INIT_KD=False
DEBUG=False


CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name arc_challenge \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/arc_challenge \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name arc_easy \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/arc_easy \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name openbookqa \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/openbookqa \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name piqa \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/piqa \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name race \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/race \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name sciq \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/sciq \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name web_questions \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/web_questions \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name wikitext \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/wikitext \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python federatedscope/main.py \
  --cfg $CFG \
  data.dataset_name hellaswag \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
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
  outdir $OUT/hellaswag \
  data.is_debug $DEBUG \

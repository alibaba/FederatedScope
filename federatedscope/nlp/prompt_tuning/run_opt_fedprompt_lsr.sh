set -e

DEVICE=$1
USE_FP16=True
CFG=baseline/config_freeze.yaml
KD_CFG=baseline/config_init_kd.yaml
GLOBAL_CFG=baseline/config_global.2.yaml
OUT=exp/opt-1.3b/fedprompt_lsr/
MODEL_TYPE=facebook/opt-1.3b
BATCH_SIZE=16
GRAD_ACCUM=1
ROUND_NUM=50
LOCAL_STEPS=10
MAX_SEQ_LEN=1024
SERVER_PREFIX_LEN=40
CLIENT_PREFIX_LEN=40
NUM_CLIENT=10
NUM_SERVER_LAYERS=24
NUM_CLIENT_LAYERS=24
CLIENT_START_LAYER_ID=0
NUM_CLIENT_LAYERS_PER_CELL=1
LR=5e-3
EPS=1e-4
NON_IID_SPLIT=False
MAKE_GLOBAL_TRAIN=True
SHARE_CLIENT_LAYER_PARAM=False
PL_INIT_KD=False
USE_PREFIX_PRJ=False
DEBUG=False


CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name arc_challenge \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/arc_challenge \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name arc_easy \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/arc_easy \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name openbookqa \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/openbookqa \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name web_questions \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/web_questions \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name hellaswag \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/hellaswag \
  data.is_debug $DEBUG \

BATCH_SIZE=1
GRAD_ACCUM=16
CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name piqa \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/piqa \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name sciq \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/sciq \
  data.is_debug $DEBUG \

CUDA_VISIBLE_DEVICES=$DEVICE python ../../main.py \
  --cfg $CFG \
  data.dataset_name race \
  data.non_iid_split $NON_IID_SPLIT \
  data.batch_size $BATCH_SIZE \
  data.max_seq_len $MAX_SEQ_LEN \
  grad.grad_accum_count $GRAD_ACCUM \
  federate.client_num $NUM_CLIENT \
  federate.total_round_num $ROUND_NUM \
  federate.make_global_train $MAKE_GLOBAL_TRAIN \
  federate.pl_init_kd $PL_INIT_KD \
  federate.pl_kd_cfg_file $KD_CFG \
  federate.pl_global_cfg_file $GLOBAL_CFG \
  model.use_fp16 $USE_FP16 \
  model.model_type $MODEL_TYPE \
  model.use_prefix_prj $USE_PREFIX_PRJ \
  model.server_prefix_len $SERVER_PREFIX_LEN \
  model.client_prefix_len $CLIENT_PREFIX_LEN \
  model.num_server_layers $NUM_SERVER_LAYERS \
  model.num_client_layers $NUM_CLIENT_LAYERS \
  model.share_client_layer_param $SHARE_CLIENT_LAYER_PARAM \
  model.client_start_layer_id $CLIENT_START_LAYER_ID \
  model.num_client_layers_per_cell $NUM_CLIENT_LAYERS_PER_CELL \
  train.optimizer.lr $LR \
  train.optimizer.eps $EPS \
  train.local_update_steps $LOCAL_STEPS \
  outdir $OUT/race \
  data.is_debug $DEBUG \

set -e

cd ../../..

echo "Run pcfednlp."

python federatedscope/main.py \
  --cfg federatedscope/nlp/baseline/config_pcfednlp.yaml \
  --client_cfg federatedscope/nlp/baseline/config_client_pcfednlp.yaml \
  federate.hfl_load_from exp/pfednlp/pretrain/ckpt/ \
  outdir exp/pcfednlp/train/ \

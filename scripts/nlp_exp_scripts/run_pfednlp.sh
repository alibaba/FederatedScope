set -e

cd ..

echo "Run pfednlp."

python federatedscope/main.py \
  --cfg federatedscope/nlp/baseline/config_pfednlp.yaml \
  --client_cfg federatedscope/nlp/baseline/config_client_pfednlp.yaml \
  federate.hfl_load_from exp/pfednlp/pretrain/ckpt/ \
  outdir exp/pfednlp/train/ \

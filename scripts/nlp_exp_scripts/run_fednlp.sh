set -e

cd ..

echo "Run fednlp."

python federatedscope/main.py \
  --cfg federatedscope/nlp/baseline/config_fednlp.yaml \
  --client_cfg federatedscope/nlp/baseline/config_client_fednlp.yaml \
  outdir exp/fednlp/train/ \

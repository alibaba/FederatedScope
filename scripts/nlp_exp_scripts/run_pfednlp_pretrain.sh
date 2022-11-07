set -e

cd ..

echo "Run pfednlp pretrain."

python federatedscope/main.py \
  --cfg federatedscope/nlp/baseline/config_pfednlp_pretrain.yaml \
  outdir exp/pfednlp/pretrain/ \

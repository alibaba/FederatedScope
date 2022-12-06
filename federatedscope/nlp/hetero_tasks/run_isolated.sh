set -e

cd ../../..

echo "Run isolated."

python federatedscope/main.py \
  --cfg federatedscope/nlp/baseline/config_isolated.yaml \
  --client_cfg federatedscope/nlp/baseline/config_client_isolated.yaml \
  outdir exp/isolated/train/ \

set -e

cd ../..

echo "Run fedopt on synthetic."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/fedavg_lr_on_synthetic.yaml \
  fedprox.use True \
  fedprox.mu 0.1

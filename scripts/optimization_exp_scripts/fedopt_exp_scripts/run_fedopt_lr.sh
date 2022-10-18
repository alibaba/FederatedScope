set -e

cd ../..

echo "Run fedopt on synthetic."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/fedavg_lr_on_synthetic.yaml \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.optimizer.lr 1. \

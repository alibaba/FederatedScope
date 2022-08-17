set -e

cd ../..

echo "Run fedopt on shakespeare."

python federatedscope/main.py --cfg federatedscope/nlp/baseline/fedavg_lstm_on_shakespeare.yaml \
  fedopt.use True \
  federate.method FedOpt \
  fedopt.optimizer.lr 1.

set -e

cd ../..

echo "Run fedopt on femnist."

python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml\
                                fedopt.use True \
                                federate.method FedOpt \
                                fedopt.optimizer.lr 1. \

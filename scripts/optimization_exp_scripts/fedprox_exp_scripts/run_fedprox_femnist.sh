set -e

cd ../..

echo "Run fedprox on femnist."

python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml \
  fedprox.use True \
  fedprox.mu 0.1

## Cross-Backend Federated Learning

We provide an example for constructing cross-backend (Tensorflow and PyTorch) federated learning, which trains an LR model on the synthetic toy data.

The server runs with Tensorflow, and clients run with PyTorch (the suggested version of Tensorflow is 1.12.0):
```shell script
# Generate toy data
python ../../scripts/distributed_scripts/gen_data.py
# Server
python ../main.py --cfg distributed_tf_server.yaml

# Clients
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_client_3.yaml
```

One of the client runs with Tensorflow, and the server and other clients run with PyTorch:
```shell script
# Generate toy data
python ../../scripts/distributed_scripts/gen_data.py
# Server
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_server.yaml

# Clients with Pytorch
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_client_1.yaml
python ../main.py --cfg ../../scripts/distributed_scripts/distributed_configs/distributed_client_2.yaml
# Clients with Tensorflow
python ../main.py --cfg distributed_tf_client_3.yaml
```

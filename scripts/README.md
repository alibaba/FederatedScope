## Scripts for Reproduction
We provide some scripts for reproducing existing algorithms with FederatedScope, which are constantly being updated.
We greatly appreciate any [contribution](https://federatedscope.io/docs/contributor/) to FederatedScope!

- [Distribute Mode](#distribute-mode)
- [Asynchronous Training Strategy](#asynchronous-training-strategy)
- [Graph Federated Learning](#graph-federated-learning)

### Distribute Mode
Users can train an LR on generated toy data with distribute mode via:
```shell script
bash distributed_scripts/run_distributed_lr.sh 
```
The FL course consists of 1 server and 3 clients, which executes on one device as simulation. Each client owns private data and the server holds a test set for global evaluation.
- For running with multiple devices, you need to specify the host/port of IP addresses in the configurations (i.e., the yaml files) and make sure these devices are connected.
Then you can launch the participants (i.e., `python federatedscope/main.py --cfg xxx.yaml`) on each provided device (Remember to launch the server first).
- For the situation that server doesn't own data and the evaluation is performed at clients, use `distributed_server_no_data.yaml` at this [line](https://github.com/alibaba/FederatedScope/blob/master/scripts/distributed_scripts/run_distributed_lr.sh#L11).

Also, users can run distribute mode with other provided datasets and models. Take training ConvNet on FEMNIST as an example:
```shell script
bash distributed_scripts/run_distributed_conv_femnist.sh 
```

### Federated Learning in Computer Vision (FL-CV)
We provide several configurations (yaml files) as examples to demonstrate how to apply FL in CV with FederatedScope.
Users can run the following comments for reproducing, and modify/add the yaml file for customization, such as using provided/customized datasets and models, tunning hyperparameters, etc.

Train ConvNet on FEMNIST with vanilla FedAvg:
```shell script
cd ..
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml
# or 
# python federatedscope/main.py --cfg scripts/example_configs/femnist.yaml
```

Train ConvNet on CelebA with vanilla FedAvg:
```shell script
cd ..
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_celeba.yaml
```

Train ConvNet on FEMNIST with FedBN:
```shell script
cd ..
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedbn_convnet2_on_femnist.yaml
```

### Asynchronous Training Strategy
We provide an example for training ConvNet on CIFAR-10 with asynchronous training strategies:
```shell script
cd ..
python federatedscope/main.py --cfg scritpes/example_configs/asyn_cifar10.yaml
```
The FL courses consists of 1 server and 200 clients, which applies `goal_achieved` strategies and set the `min_received_num=10` and `staleness_toleration=10`.
Users can change the configurations related to asynchronous training for customization. Please see [configurations](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/core/configs).

### Graph Federated Learning
Please refer to [gfl](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/gfl) for more details.

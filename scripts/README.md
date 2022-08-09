## Scripts for Reproduction
We provide some scripts for reproducing existing algorithms with FederatedScope, which are constantly being updated.
We greatly appreciate any [contribution](https://federatedscope.io/docs/contributor/) to FederatedScope!

General:
- [Distribute Mode](#distribute-mode)

### Distribute Mode
Users can train an LR on generated toy data with distribute mode via (`pwd=FederatedScope/`):
```shell script
bash scripts/run_distributed_lr.sh 
```
The FL course consists of 1 server and 3 clients, which executes on one device as simulation. Each client owns private data and the server holds a test set for global evaluation.
- For running with multiple devices, you need to specify the host/port of IP addresses in the configurations (i.e., the yaml files) and make sure these devices are connected.
Then you can launch the participants (i.e., `python federatedscope/main.py --cfg xxx.yaml`) on each provided device (Remember to launch the server first).
- For the situation that server doesn't own data and the evaluation is performed at clients, use `distributed_server_no_data.yaml` at this [line](https://github.com/alibaba/FederatedScope/blob/master/scripts/run_distributed_lr.sh#L11).

Also, users can run distribute mode with other provided datasets and models. Take training ConvNet on FEMNIST as an example:
```shell script
bash scripts/run_distributed_conv_femnist.sh 
```

### Federated Learning in Computer Vision (FL-CV)
We provide several configurations (yaml files) as examples to demonstrate how to apply FL in CV with FederatedScope.
Users can run the following comments for reproducing, and modify/add the yaml file for customization, such as using provided/customized datasets and models, tunning hyperparameters, etc.

Train convent-2 on FEMNIST with vanilla FedAvg (`pwd=FederatedScope/`):
```shell script
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml
# or 
# python federatedscope/main.py --cfg federatedscope/example_configs/femnist.yaml
```

Train convent-2 on CelebA with vanilla FedAvg (`pwd=FederatedScope/`):
```shell script
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_celeba.yaml
```

Train convent-2 on FEMNIST with FedBN (`pwd=FederatedScope/`):
```shell script
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedbn_convnet2_on_femnist.yaml
```

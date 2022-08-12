## Scripts for Reproduction
We provide some scripts for reproducing existing algorithms with FederatedScope, which are constantly being updated.
We greatly appreciate any [contribution](https://federatedscope.io/docs/contributor/) to FederatedScope!

- [Distribute Mode](#distribute-mode)
- [Asynchronous Training Strategy](#asynchronous-training-strategy)
- [Graph Federated Learning](#graph-federated-learning)
- [Attacks in Federated Learning](#attacks-in-federated-learning)

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
python federatedscope/main.py --cfg scripts/example_configs/asyn_cifar10.yaml
```
The FL courses consists of 1 server and 200 clients, which applies `goal_achieved` strategies and set the `min_received_num=10` and `staleness_toleration=10`.
Users can change the configurations related to asynchronous training for customization. Please see [configurations](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/core/configs).

Note that users can manually download [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and put it to `FederatedScope/data` if the automatic download process failed. And for `resource_info_file`, we take the [client_device_capacity](https://github.com/SymbioticLab/FedScale/blob/master/benchmark/dataset/data/device_info/client_device_capacity) provided by [1] as an example.

### Graph Federated Learning
Please refer to [gfl](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/gfl) for more details.

### Attacks in Federated Learning

#### Privacy Attacks
We provide the following four examples to run the membership inference attack, property inference attack, class representative attack and training data/label inference attack, respectively. 

Membership inference attack:

Run the attack in [2]:
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/gradient_ascent_MIA_on_femnist.yaml
```

Property inference attack: Run the BPC [2] attack
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/PIA_toy.yaml
```

Class representative attack: Run DCGAN [3] attack
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/CRA_fedavg_convnet2_on_femnist.yaml
```

Training data/label inference attack: Run the DLG [4] attack 
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/reconstruct_fedavg_opt_on_femnist.yaml
```


#### Backdoor Attacks

Run the BadNets [5] attack:
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/backdoor_attack/backdoor_badnet_fedavg_convnet2_on_femnist.yaml
```

### References:  
[1] Lai F, Dai Y, Singapuram S, et al. "FedScale: Benchmarking model and system performance of federated learning at scale." International Conference on Machine Learning. PMLR, 2022: 11814-11827.

[2] Nasr, Milad, R. Shokri and Amir Houmansadr. "Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks." ArXiv abs/1812.00910 (2018).

[3] Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security.

[4] Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).

[5] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2019. "BadNets: Evaluating Backdooring Attacks on Deep Neural Networks." IEEE Access 7 (2019), 47230-47244.

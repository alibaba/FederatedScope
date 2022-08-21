## Scripts for Reproduction
We provide some scripts for reproducing existing algorithms with FederatedScope, which are constantly being updated.
We greatly appreciate any [contribution](https://federatedscope.io/docs/contributor/) to FederatedScope!

- [Distribute Mode](#distribute-mode)
- [Asynchronous Training Strategy](#asynchronous-training-strategy)
- [Graph Federated Learning](#graph-federated-learning)
- [Attacks in Federated Learning](#attacks-in-federated-learning)
- [Federated Optimization Algorithm](#federated-optimization-algorithm)
- [Personalized Federated Learning](#personalized-federated-learning)
- [Differential Privacy in Federated Learning](#differential-privacy-in-federated-learning)
- [Matrix Factorization in Federated Learning](#matrix-factorization-in-federated-learning)

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

- The case that the target data are not in the training set:
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/gradient_ascent_MIA_on_femnist.yaml
```

- The case that the target data are not in the training set:
```shell script
python federatedscope/main.py --cfg scripts/attack_exp_scripts/privacy_attack/gradient_ascent_MIA_on_femnist_simu_in.yaml
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

### Federated Optimization Algorithm
Users can replace the fedavg algorithm by other federated optimization algorithms.
In the following we provide some running scripts for FedOpt[6] and FedProx[7] on different dataset.

#### FedOpt
Run fedopt on different dataset via
```bash
# on femnist
bash optimization_exp_scripts/fedopt_exp_scripts/run_fedopt_femnist.sh
# on synthetic
bash optimization_exp_scripts/fedopt_exp_scripts/run_fedopt_lr.sh
# on shakespeare
bash optimization_exp_scripts/fedopt_exp_scripts/run_fedopt_shakespeare.sh
```

#### FedProx
Run fedprox on different dataset via
```bash
# on femnist
bash optimization_exp_scripts/fedprox_exp_scripts/run_fedprox_femnist.sh
# on lr
bash optimization_exp_scripts/fedprox_exp_scripts/run_fedprox_lr.sh
# on shakespeare
bash optimization_exp_scripts/fedprox_exp_scripts/run_fedprox_shakespeare.sh
```

### Personalized Federated Learning
Users can replace the fedavg 
algorithm by other personalized federated learning algorithms.
In the following we provide some running scripts for FedBN [9], Ditto [10], 
pFedMe [11], and FedEM [12] on several datasets. More running examples for 
other personalized FL 
methods and datasets can be found in our [benchemark](https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench).

#### FedBN
To use FedBN, we can specify the local parameter names related to BN as 
`cfg.personalization.local_param=['bn']`. We can run FedBN via:
```bash
cd personalization_exp_scripts
# on femnist
bash run_femnist_fedbn.sh
```

#### Ditto
To use Ditto, we can specify cfg as `federate.method=ditto` and determine the
regularization value such as `personalization.regular_weight=0.1`.
We can run Ditto on different dataset as follows:
```bash
cd personalization_exp_scripts
# on femnist
bash run_femnist_ditto.sh
# on lr
bash run_synthetic_ditto.sh
# on shakespeare
bash run_shakespeare_ditto.sh
```

#### pFedMe 
To use pFedMe, we can specify cfg as `federate.method=pFedMe` and determine 
its hyper-parameters such as `personalization.lr=0.1`, 
`personalization.beta=1.0` and `personalization.K=3`.
We can run pFedMe on different dataset via:
```bash
cd personalization_exp_scripts
# on femnist
bash run_femnist_pfedme.sh
# on lr
bash run_synthetic_pfedme.sh
# on shakespeare
bash run_shakespeare_pfedme.sh
```

#### FedEM 
To use FedEM, we can specify cfg as `federate.method=FedEM` and determine
its hyper-parameters such as `model.model_num_per_trainer=3`.
We can run FedEM on different dataset as follows:
```bash
cd personalization_exp_scripts
# on femnist
bash run_femnist_fedem.sh
# on lr
bash run_synthetic_fedem.sh
# on shakespeare
bash run_shakespeare_fedem.sh
```


### Differential Privacy in Federated Learning

Users can train models with protection of differential privacy. 
Taking the dataset FEMNIST as an example, execute the running scripts via:
```bash
bash dp_exp_scripts/run_femnist_dp_standalone.sh
```
You can also enable DP algorithm with other dataset and models by adding the following configurations:
```yaml
nbafl: 
  use: True
  mu: 0.1
  epsilon: 10
  constant: 30
  w_clip: 0.1
federate:
  join_in_info: ["num_sample"]
```

### Matrix Factorization in Federated Learning
We support federated matrix factorization tasks in both vertical and horizontal federated learning scenario. 
Users can run matrix factorization tasks on MovieLen dataset via
```bash
# vfl
bash mf_exp_scripts/run_movielens1m_vfl_standalone.sh
# hfl
bash mf_exp_scripts/run_movielens1m_hfl_standalone.sh
```
Also, we support SGDMF[8] algorithm in federated learning, and users can run it via
```bash
# hfl
bash mf_exp_scripts/run_movielens1m_hflsgdmf_standalone.sh
# vfl
bash mf_exp_scripts/run_movielens1m_vflsgdmf_standalone.sh
```

#### References:
[1] Lai F, Dai Y, Singapuram S, et al. "FedScale: Benchmarking model and system performance of federated learning at scale." International Conference on Machine Learning. PMLR, 2022: 11814-11827.

[2] Nasr, Milad, R. Shokri and Amir Houmansadr. "Comprehensive Privacy Analysis of Deep Learning: Stand-alone and Federated Learning under Passive and Active White-box Inference Attacks." ArXiv abs/1812.00910 (2018).

[3] Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz. "Deep models under the GAN: information leakage from collaborative deep learning." Proceedings of the 2017 ACM SIGSAC conference on computer and communications security.

[4] Zhu, Ligeng, Zhijian Liu, and Song Han. "Deep leakage from gradients." Advances in Neural Information Processing Systems 32 (2019).

[5] Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. 2019. "BadNets: Evaluating Backdooring Attacks on Deep Neural Networks." IEEE Access 7 (2019), 47230-47244.

[6] Asad M, Moustafa A, Ito T. "FedOpt: Towards communication efficiency and privacy preservation in federated learning". Applied Sciences, 2020, 10(8): 2864.

[7] Anit Kumar Sahu, Tian Li, Maziar Sanjabi, Manzil Zaheer, Ameet Talwalkar, Virginia Smith. "On the Convergence of Federated Optimization in Heterogeneous Networks." ArXiv abs/1812.06127 (2018).

[8] Zitao Li, Bolin Ding, Ce Zhang, Ninghui Li, Jingren Zhou. "Federated Matrix Factorization with Privacy Guarantee." Proceedings of the VLDB Endowment, 15(4): 900-913 (2021).

[9] Li, Xiaoxiao, et al. “Fedbn: Federated learning on non-iid features via local batch normalization.” arXiv preprint arXiv:2102.07623 (2021).

[10] Li, Tian, et al. “Ditto: Fair and robust federated learning through personalization.” International Conference on Machine Learning. PMLR, 2021.

[11] T Dinh, Canh, Nguyen Tran, and Josh Nguyen. “Personalized federated learning with moreau envelopes.” Advances in Neural Information Processing Systems 33 (2020): 21394-21405.

[12] Marfoq, Othmane, et al. “Federated multi-task learning under a mixture of distributions.” Advances in Neural Information Processing Systems 34 (2021).
<h1 align="center">
    <img src="https://img.alicdn.com/imgextra/i4/O1CN01yp6zdb23HOJJkCmZg_!!6000000007230-2-tps-2048-1009.png" width="400" alt="federatedscope-logo">
</h1>

![](https://img.shields.io/badge/language-python-blue.svg)
![](https://img.shields.io/badge/license-Apache-000000.svg)
[![Website](https://img.shields.io/badge/website-FederatedScope-0000FF)](https://federatedscope.io/)
[![Playground](https://shields.io/badge/JupyterLab-Enjoy%20Your%20FL%20Journey!-F37626?logo=jupyter)](https://try.federatedscope.io/)
[![Contributing](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://federatedscope.io/docs/contributor/)

FederatedScope is a comprehensive federated learning platform that provides convenient usage and flexible customization for various federated learning tasks in both academia and industry.  Based on an event-driven architecture, FederatedScope integrates rich collections of functionalities to satisfy the burgeoning demands from federated learning, and aims to build up an easy-to-use platform for promoting learning safely and effectively.

A detailed tutorial is provided on our [website](https://federatedscope.io/).

## News
- [07-11-2022] We are hosting the CIKM'22 AnalytiCup competition. For more details, please see [competition website](https://federatedscope.io/competition/).
- [06-17-2022] We release **pFL-Bench**, a comprehensive benchmark for personalized Federated Learning (pFL), containing 10+ datasets and 20+ baselines. [[code](https://github.com/alibaba/FederatedScope/tree/master/benchmark/pFL-Bench), [pdf](https://arxiv.org/abs/2206.03655)]
- [06-17-2022] We release **FedHPO-B**, a benchmark suite for studying federated hyperparameter optimization. [[code](https://github.com/alibaba/FederatedScope/tree/master/benchmark/FedHPOB), [pdf](https://arxiv.org/abs/2206.03966)]
- [06-17-2022] We release **B-FHTL**, a benchmark suit for studying federated hetero-task learning. [[code](https://github.com/alibaba/FederatedScope/tree/master/benchmark/B-FHTL), [pdf](https://arxiv.org/abs/2206.03436)]
- [06-13-2022] Our project was receiving an attack, which has been resolved. [More details](https://github.com/alibaba/FederatedScope/blob/master/doc/news/06-13-2022_Declaration_of_Emergency.txt).
- [05-25-2022] Our paper [FederatedScope-GNN](https://arxiv.org/abs/2204.05562) has been accepted by KDD'2022!
- [05-06-2022] We release FederatedScope v0.1.0! 

## Quick Start

We provide an end-to-end example for users to start running a standard FL course with FederatedScope.

### Step 1. Installation

First of all, users need to clone the source code and install the required packages (we suggest python version >= 3.9). You can choose between the following two installation methods (via docker or conda) to install FederatedScope.

```bash
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope
```
#### Use Docker

You can build docker image and run with docker env (cuda 11 and torch 1.10):

```
docker build -f environment/docker_files/federatedscope-torch1.10.Dockerfile -t alibaba/federatedscope:base-env-torch1.10 .
docker run --gpus device=all --rm -it --name "fedscope" -w $(pwd) alibaba/federatedscope:base-env-torch1.10 /bin/bash
```
If you need to run with down-stream tasks such as graph FL, change the requirement/docker file name into another one when executing the above commands:
```
# environment/requirements-torch1.10.txt -> 
environment/requirements-torch1.10-application.txt

# environment/docker_files/federatedscope-torch1.10.Dockerfile ->
environment/docker_files/federatedscope-torch1.10-application.Dockerfile
```
Note: You can choose to use cuda 10 and torch 1.8 via changing `torch1.10` to `torch1.8`.
The docker images are based on the nvidia-docker. Please pre-install the NVIDIA drivers and `nvidia-docker2` in the host machine. See more details [here](https://github.com/alibaba/FederatedScope/tree/master/environment/docker_files).

#### Use Conda

We recommend using a new virtual environment to install FederatedScope:

```bash
conda create -n fs python=3.9
conda activate fs
```

If your backend is torch, please install torch in advance ([torch-get-started](https://pytorch.org/get-started/locally/)). For example, if your cuda version is 11.3 please execute the following command:

```bash
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

For users with Apple M1 chips:
```bash
conda install pytorch torchvision torchaudio -c pytorch
# Downgrade torchvision to avoid segmentation fault
python -m pip install torchvision==0.11.3
```

Finally, after the backend is installed, you can install FederatedScope from `source`:

##### From source

```bash
python setup.py install

# Or (for dev mode)
pip install -e .[dev]
pre-commit install
```

Now, you have successfully installed the minimal version of FederatedScope. (**Optinal**) For application version including graph, nlp and speech, run:

```bash
bash environment/extra_dependencies_torch1.10-application.sh
```

### Step 2. Prepare datasets

To run an FL task, users should prepare a dataset. 
The DataZoo provided in FederatedScope can help to automatically download and preprocess widely-used public datasets for various FL applications, including CV, NLP, graph learning, recommendation, etc. Users can directly specify `cfg.data.type = DATASET_NAME`in the configuration. For example, 

```bash
cfg.data.type = 'femnist'
```

To use customized datasets, you need to prepare the datasets following a certain format and register it. Please refer to [Customized Datasets](https://federatedscope.io/docs/own-case/#data) for more details.

### Step 3. Prepare models

Then, users should specify the model architecture that will be trained in the FL course.
FederatedScope provides a ModelZoo that contains the implementation of widely adopted model architectures for various FL applications. Users can set up `cfg.model.type = MODEL_NAME` to apply a specific model architecture in FL tasks. For example,

```yaml
cfg.model.type = 'convnet2'
```

FederatedScope allows users to use customized models via registering. Please refer to [Customized Models](https://federatedscope.io/docs/own-case/#model) for more details about how to customize a model architecture.

### Step 4. Start running an FL task

Note that FederatedScope provides a unified interface for both standalone mode and distributed mode, and allows users to change via configuring. 

#### Standalone mode

The standalone mode in FederatedScope means to simulate multiple participants (servers and clients) in a single device, while participants' data are isolated from each other and their models might be shared via message passing. 

Here we demonstrate how to run a standard FL task with FederatedScope, with setting `cfg.data.type = 'FEMNIST'`and `cfg.model.type = 'ConvNet2'` to run vanilla FedAvg for an image classification task. Users can customize training configurations, such as `cfg.federated.total_round_num`, `cfg.data.batch_size`, and `cfg.train.optimizer.lr`, in the configuration (a .yaml file), and run a standard FL task as: 

```bash
# Run with default configurations
python federatedscope/main.py --cfg federatedscope/example_configs/femnist.yaml
# Or with custom configurations
python federatedscope/main.py --cfg federatedscope/example_configs/femnist.yaml federated.total_round_num 50 data.batch_size 128
```

Then you can observe some monitored metrics during the training process as:

```
INFO: Server has been set up ...
INFO: Model meta-info: <class 'federatedscope.cv.model.cnn.ConvNet2'>.
... ...
INFO: Client has been set up ...
INFO: Model meta-info: <class 'federatedscope.cv.model.cnn.ConvNet2'>.
... ...
INFO: {'Role': 'Client #5', 'Round': 0, 'Results_raw': {'train_loss': 207.6341676712036, 'train_acc': 0.02, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.152683353424072}}
INFO: {'Role': 'Client #1', 'Round': 0, 'Results_raw': {'train_loss': 209.0940284729004, 'train_acc': 0.02, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1818805694580075}}
INFO: {'Role': 'Client #8', 'Round': 0, 'Results_raw': {'train_loss': 202.24929332733154, 'train_acc': 0.04, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.0449858665466305}}
INFO: {'Role': 'Client #6', 'Round': 0, 'Results_raw': {'train_loss': 209.43883895874023, 'train_acc': 0.06, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1887767791748045}}
INFO: {'Role': 'Client #9', 'Round': 0, 'Results_raw': {'train_loss': 208.83140087127686, 'train_acc': 0.0, 'train_total': 50, 'train_loss_regular': 0.0, 'train_avg_loss': 4.1766280174255375}}
INFO: ----------- Starting a new training round (Round #1) -------------
... ...
INFO: Server: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 163.029045
... ...
INFO: Server: Final evaluation is finished! Starting merging results.
... ...
```

#### Distributed mode

The distributed mode in FederatedScope denotes running multiple procedures to build up an FL course, where each procedure plays as a participant (server or client) that instantiates its model and loads its data. The communication between participants is already provided by the communication module of FederatedScope.

To run with distributed mode, you only need to:

- Prepare isolated data file and set up `cfg.distribute.data_file = PATH/TO/DATA` for each participant;
- Change `cfg.federate.model = 'distributed'`, and specify the role of each participant  by `cfg.distributed.role = 'server'/'client'`.
- Set up a valid address by `cfg.distribute.host = x.x.x.x` and `cfg.distribute.port = xxxx`. (Note that for a server, you need to set up server_host/server_port for listening messge, while for a client, you need to set up client_host/client_port for listening and server_host/server_port for sending join-in applications when building up an FL course)

We prepare a synthetic example for running with distributed mode:

```bash
# For server
python main.py --cfg federatedscope/example_configs/distributed_server.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx

# For clients
python main.py --cfg federatedscope/example_configs/distributed_client_1.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
python main.py --cfg federatedscope/example_configs/distributed_client_2.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
python main.py --cfg federatedscope/example_configs/distributed_client_3.yaml distribute.data_file 'PATH/TO/DATA' distribute.server_host x.x.x.x distribute.server_port xxxx distribute.client_host x.x.x.x distribute.client_port xxxx
```

An executable example with generated toy data can be run with:
```bash
# Generate the toy data
python scripts/gen_data.py

# Firstly start the server that is waiting for clients to join in
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_server.yaml distribute.data_file toy_data/server_data distribute.server_host 127.0.0.1 distribute.server_port 50051

# Start the client #1 (with another process)
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_client_1.yaml distribute.data_file toy_data/client_1_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50052
# Start the client #2 (with another process)
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_client_2.yaml distribute.data_file toy_data/client_2_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50053
# Start the client #3 (with another process)
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_client_3.yaml distribute.data_file toy_data/client_3_data distribute.server_host 127.0.0.1 distribute.server_port 50051 distribute.client_host 127.0.0.1 distribute.client_port 50054
```

And you can observe the results as (the IP addresses are anonymized with 'x.x.x.x'):

```
INFO: Server: Listen to x.x.x.x:xxxx...
INFO: Server has been set up ...
Model meta-info: <class 'federatedscope.core.lr.LogisticRegression'>.
... ...
INFO: Client: Listen to x.x.x.x:xxxx...
INFO: Client (address x.x.x.x:xxxx) has been set up ...
Client (address x.x.x.x:xxxx) is assigned with #1.
INFO: Model meta-info: <class 'federatedscope.core.lr.LogisticRegression'>.
... ...
{'Role': 'Client #2', 'Round': 0, 'Results_raw': {'train_avg_loss': 5.215108394622803, 'train_loss': 333.7669372558594, 'train_total': 64}}
{'Role': 'Client #1', 'Round': 0, 'Results_raw': {'train_total': 64, 'train_loss': 290.9668884277344, 'train_avg_loss': 4.54635763168335}}
----------- Starting a new training round (Round #1) -------------
... ...
INFO: Server: Training is finished! Starting evaluation.
INFO: Client #1: (Evaluation (test set) at Round #20) test_loss is 30.387419
... ...
INFO: Server: Final evaluation is finished! Starting merging results.
... ...
```


## Advanced

As a comprehensive FL platform, FederatedScope provides the fundamental implementation to support requirements of various FL applications and frontier studies, towards both convenient usage and flexible extension, including:

- **Personalized Federated Learning**: Client-specific model architectures and training configurations are applied to handle the non-IID issues caused by the diverse data distributions and heterogeneous system resources.
- **Federated Hyperparameter Optimization**: When hyperparameter optimization (HPO) comes to Federated Learning, each attempt is extremely costly due to multiple rounds of communication across participants. It is worth noting that HPO under the FL is unique and more techniques should be promoted such as low-fidelity HPO.
- **Privacy Attacker**: The privacy attack algorithms are important and convenient to verify the privacy protection strength of the design FL systems and algorithms, which is growing along with Federated Learning.
- **Graph Federated Learning**: Working on the ubiquitous graph data, Graph Federated Learning aims to exploit isolated sub-graph data to learn a global model, and has attracted increasing popularity.
- **Recommendation**: As a number of laws and regulations go into effect all over the world, more and more people are aware of the importance of privacy protection, which urges the recommender system to learn from user data in a privacy-preserving manner.
- **Differential Privacy**: Different from the encryption algorithms that require a large amount of computation resources,  differential privacy is an economical yet flexible technique to protect privacy, which has achieved great success in database and is ever-growing in federated learning.
- ...

More supports are coming soon! We have prepared a [tutorial](https://federatedscope.io/) to provide more details about how to utilize FederatedScope to enjoy your journey of Federated Learning!

Materials of related topics are constantly being updated, please refer to [FL-Recommendation](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/FL-Recommendation), [Federated-HPO](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/Federated_HPO), [Personalized FL](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/Personalized_FL), [Federated Graph Learning](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/Federated_Graph_Learning), [FL-NLP](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/FL-NLP), [FL-Attacker](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/FL-Attacker), [FL-Incentive-Mechanism](https://github.com/alibaba/FederatedScope/tree/master/materials/paper_list/FL-Incentive) and so on. 

## Documentation

The classes and methods of FederatedScope have been well documented so that users can generate the API references by:

```shell
pip install -r requirements-doc.txt
make html
```

We put the API references on our [website](https://federatedscope.io/refs/index).

## License

FederatedScope is released under Apache License 2.0.

## Publications
If you find FederatedScope useful for your research or development, please cite the following <a href="https://arxiv.org/abs/2204.05011" target="_blank">paper</a>:
```
@article{federatedscope,
  title = {FederatedScope: A Flexible Federated Learning Platform for Heterogeneity},
  author = {Xie, Yuexiang and Wang, Zhen and Chen, Daoyuan and Gao, Dawei and Yao, Liuyi and Kuang, Weirui and Li, Yaliang and Ding, Bolin and Zhou, Jingren},
  journal={arXiv preprint arXiv:2204.05011},
  year = {2022},
}
```
More publications can be found in the [Publications](https://federatedscope.io/year-archive/).

## Contributing

We **greatly appreciate** any contribution to FederatedScope! We provide a developer version of FederatedScope with additional pre-commit hooks to perform commit checks compared to the official version:

```bash
# Install the developer version
pip install -e .[dev]
pre-commit install

# Or switch to the developer version from the official version
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

You can refer to [Contributing to FederatedScope](https://federatedscope.io/docs/contributor/) for more details.

Welcome to join in our [Slack channel](https://join.slack.com/t/federatedscopeteam/shared_invite/zt-1apmfjqmc-hvpYbsWJdm7D93wPNXbqww), or DingDing group (please scan the following QR code) for discussion.

<img width="150" src="https://img.alicdn.com/imgextra/i4/O1CN01heXHpf1zuXhcOCgGF_!!6000000006774-2-tps-860-861.png" width="400" alt="federatedscope-logo">

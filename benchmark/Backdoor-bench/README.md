# Benchmark for Back-door Attack on Personalized Federated Learning

Back-door-bench is a benchmark for back-door attacks on personalized
 federated learning. It contains backdoor attacks including edge, BadNet
 , blended and SIG. The attacked pFL methods include: FedAvg
 , Fintuning (FT), Ditto, FedEM, pFedMe, FedBN, FedRep. More details about
  the benchmark settings and experimental results refer to [paper](https://arxiv.org/pdf/2302.01677.pdf). 

**Notice**:
Considering FederatedScope is an open-sourced library that updates
 frequently, to ensure the reproducibility of the experimental results, 
we create a new branch `backdoor-bench`. The users can reproduce the results by
 running the configs under the directory [`scripts/B-backdoor_scripts
 /attack_config`](https://github.com/alibaba/FederatedScope/tree/backdoor-bench/scripts/backdoor_scripts/attack_config). The results of our paper []() is located
  in `paper_plot/results_all`.


## Quick Start

To run the script, you should 
- First clone the repository [FederatedScope](https://github.com/alibaba/FederatedScope),
- Then follow [README.md](https://github.com/alibaba/FederatedScope/blob/master/README.md) to build the running environment for FederatedScope, 
- Switch to the branch `backdoor-bench` and run the scripts
```bash
# Step-1. clone the repository 
git clone https://github.com/alibaba/FederatedScope.git

# Step-2. follow https://github.com/alibaba/FederatedScope/blob/master/README.md to build the running environment

# Step-3. install packages required by the benchmark
pip install opencv-python matplotlib pympler scikit-learn

# Step-3. switch to the branch `backdoor-bench` for the benchmark
git fetch
git switch backdoor-bench

# Step-4. run the baseline (taking attacking FedAvg with Edge type trigger
 as an example)
cd FederatedScope
python federatedscope/main.py --cfg scripts/backdoor_scripts/attack_config/backdoor_fedavg_resnet18_on_cifar10_small.yaml

```


## Publications

If you find Back-door-bench useful for your research or development, please
 cite the following [paper](https://arxiv.org/pdf/2302.01677.pdf):

```tex
@article{DBLP:journals/corr/abs-2302-01677,
  author       = {Zeyu Qin and Liuyi Yao and Daoyuan Chen and Yaliang Li and
                  Bolin Ding and Minhao Cheng},
  title        = {Revisiting Personalized Federated Learning: Robustness
 Against Backdoor Attacks},
  journal      = {CoRR},
  volume       = {abs/2302.01677},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2302.01677},
  doi          = {10.48550/arXiv.2302.01677},
  eprinttype    = {arXiv},
}
```
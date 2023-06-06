# Benchmark for Back-door Attack on Personalized Federated Learning



Backdoor-bench is a benchmark for backdoor attacks on personalized federated learning. It contains backdoor attacks including [edge-based trigger](https://arxiv.org/abs/2007.05084), [BadNet](https://ieeexplore.ieee.org/document/8685687), [Blended](https://arxiv.org/abs/1712.05526) and [SIG](https://arxiv.org/abs/1902.11237). The attacked pFL methods include: FedAvg, Fine-tuning (FT), Ditto, FedEM, pFedMe, FedBN, FedRep. More details about the benchmark settings and experimental results refer to our KDD [paper](https://arxiv.org/abs/2302.01677). 

**Notice**:
Considering FederatedScope is an open-sourced library that updates frequently, to ensure the reproducibility of the experimental results, we create a new branch `backdoor-bench`. The users can reproduce the results by running the configs under the directory [scripts/B-backdoor_scripts attack_config](https://github.com/alibaba/FederatedScope/tree/backdoor-bench/scripts/backdoor_scripts/attack_config). The results of our paper is located in `paper_plot/results_all`.

## Publications

If you find Back-door-bench useful for your research or development, please cite the following [paper](https://arxiv.org/pdf/2302.01677.pdf):

```tex
@inproceedings{
qin2023revisiting,
title={Revisiting Personalized Federated Learning: Robustness Against Backdoor Attacks},
author={Zeyu Qin and Liuyi Yao and Daoyuan Chen and Yaliang Li and Bolin Ding and Minhao Cheng},
booktitle={29th SIGKDD Conference on Knowledge Discovery and Data Mining - Applied Data Science Track},
year={2023},
}
```

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

# Step-4. run the baseline (taking attacking FedAvg with Edge type trigger as an example)
cd FederatedScope
python federatedscope/main.py --cfg scripts/backdoor_scripts/attack_config/backdoor_fedavg_resnet18_on_cifar10_small.yaml

```
## Reimplementing Results of Paper

The all scripts of conducting experiments are in file [attack_config](https://github.com/alibaba/FederatedScope/tree/backdoor-bench/scripts/backdoor_scripts/attack_config). 
- **Backdoor or not**: Files with 'backdoor' in their filename are experimental instructions related to backdoor poisoning during the training process. Files without 'backdoor' are experimental instructions about normal FL or pFL training process.  
- **Models**: Files with different models name represents experiments with using different models, such as "convnet" or "resnet18".
- **Datasets**: Files with different dataset name represents experiments on different datasets, such as "femnist" or "cifar10".
- **pFL Methods**: Files with different method name represents experiments with using different pFL methods. 
- **IID vs Non-IID**: Files with 'iid' represents experiments under IID settings. 
- **Ablation Study**: Files with 'abl' represents ablation studies of pFL methods conducted in Section 5. 
- **FedBN**: Files with 'bn' and 'para' or 'sta' mean experiments of Fed-para and Fed-sta conducted in Section 5.1. 
- **Existing Defense**: Experiments about existing defense methods:
    *  Krum: please set attack.krum: True
    *  Multi-Krum: please set attack.multi_krum: True
    *  Norm_clip: please set attack.norm_clip: True and tune attack.norm_clip_value. 
    *  Adding noise: please tune attack.dp_noise. 

**Notice:** The Files with 'small' or 'avg' are about experiments with changing attackers since we wish to test whether the size of the local dataset possessed by the attacker will have an impact on the success of the backdoor poisoning. You can ignore them. 

----

## Explanations about Attack Config


    attack:
        setting: 'fix' --fix-frequency attack setting
        freq: 10 --the adversarial client is selected for every fixed 10 round.
        attack_method: 'backdoor'
        attacker_id: 15 --the client id of attacker
        label_type: 'dirty' --dirty or clean-label attacks. We now only support dirty-label attacks
        trigger_type: gridTrigger --BadNet: gridTrigger; Blended: hkTrigger; edge: edge; SIG: sigTrigger
        edge_num: 500 --the number of samples with edge trigger
        poison_ratio: 0.5 --poisoning ratio of local training dataset
        target_label_ind: 9 --target label of backdoor attacks
        self_opt: False --you can ignore it since we do not test it. 
        self_lr: 0.1 --you can ignore it since we do not test it. 
        self_epoch: 6 --you can ignore it since we do not test it. 
        scale_poisoning: False --you can ignore it since we do not test it. 
        scale_para: 3.0 --you can ignore it since we do not test it. 
        pgd_poisoning: False --you can ignore it since we do not test it. 
        mean: [0.4914, 0.4822, 0.4465] --normalizations used in backdoor attacks (different dataset have different settings.)
        std: [0.2023, 0.1994, 0.2010]




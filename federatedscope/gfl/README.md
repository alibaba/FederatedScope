# FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning

FederatedScope-GNN (FS-G) is a unified, comprehensive and efficient package for federated graph learning.  We provide a hands-on tutorial here, while for more detailed tutorial, please refer to [FGL Tutorial](https://federatedscope.io/docs/graph/).

## Quick Start

Let’s start with a two-layer GCN on FedCora to familiarize you with FS-G.

### Step 1. Installation

The installation of FS-G follows FederatedScope, please refer to [Installation](https://github.com/alibaba/FederatedScope#step-1-installation).

After installing the minimal version of FederatedScope, you should install extra dependencies ([PyG](https://github.com/pyg-team/pytorch_geometric), rdkit, and nltk) for the application version of FGL, run:

```bash
conda install -y pyg==2.0.4 -c pyg
conda install -y rdkit=2021.09.4=py39hccf6a74_0 -c conda-forge
conda install -y nltk
```

Now, you have successfully installed the FGL version of FederatedScope.

### Step 2. Start with built-in functions

TBD

## Advanced

### Start with built-in functions

Describe yaml here.

### Start with customized functions

xxx

## Reproduce the main experimental results

xxx



## Publications

If you find FS-G useful for research or development, please cite the following [paper](https://arxiv.org/abs/2204.05562):

```latex
@inproceedings{federatedscopegnn,
  title     = {FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning},
  author    = {Zhen Wang and Weirui Kuang and Yuexiang Xie and Liuyi Yao and Yaliang Li and Bolin Ding and Jingren Zhou},
  booktitle = {Proc.\ of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'22)},
  year      = {2022}
}
```


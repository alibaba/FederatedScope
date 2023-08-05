# FederatedScope-GNN: Towards a Unified, Comprehensive and Efficient Package for Federated Graph Learning

FederatedScope-GNN (FS-G) is a unified, comprehensive and efficient package for federated graph learning. We provide a hands-on tutorial here, while for more detailed tutorial, please refer to [FGL Tutorial](https://federatedscope.io/docs/graph/).

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

### Step 2. Run with exmaple config

Now, we train a two-layer GCN on FedCora with FedAvg.

```bash
python federatedscope/main.py --cfg federatedscope/gfl/baseline/example.yaml
```

For more details about customized configurations, see **Advanced**.

## Reproduce the results in our paper

We provide scripts (grid search to find optimal results) to reproduce the results of our experiments.

* Node-level tasks, please refer to `federatedscope/gfl/baseline/repro_exp/node_level/`:

  ```bash
  # Example of FedAvg
  cd federatedscope/gfl/baseline/repro_exp/node_level/
  bash run_node_level.sh 0 cora louvain
  
  # Example of FedAvg
  bash run_node_level.sh 0 cora random
  
  # Example of FedOpt
  bash run_node_level_opt.sh 0 cora louvain gcn 0.25 4
  
  # Example of FedProx
  bash run_node_level_prox.sh 0 cora louvain gcn 0.25 4
  ```

* Link-level tasks, please refer to `federatedscope/gfl/baseline/repro_exp/link_level/`:

  ```bash
  cd federatedscope/gfl/baseline/repro_exp/link_level/
  
  # Example of FedAvg
  bash run_link_level_KG.sh 0 wn18 rel_type
  
  # Example of FedOpt
  bash run_link_level_opt.sh 0 wn18 rel_type gcn 0.25 16
  
  # Example of FedProx
  bash run_link_level_prox.sh 7 wn18 rel_type gcn 0.25 16
  ```

* Graph-level tasks, please refer to `federatedscope/gfl/baseline/repro_exp/graph_level/`:

  ```bash
  cd federatedscope/gfl/baseline/repro_exp/graph_level/
  
  # Example of FedAvg
  bash run_graph_level.sh 0 proteins
  
  # Example of FedOpt
  bash run_graph_level_opt.sh 0 proteins gcn 0.25 4
  
  # Example of FedProx
  bash run_graph_level_prox.sh 0 proteins gcn 0.25 4
  ```

## Advanced

### Start with built-in functions

You can easily run through a customized `yaml` file:

```yaml
# Whether to use GPU
use_gpu: True

# Deciding which GPU to use
device: 0

# Federate learning related options
federate:
  # `standalone` or `distributed`
  mode: standalone
  # Evaluate in Server or Client test set
  make_global_eval: True
  # Number of dataset being split
  client_num: 5
  # Number of communication round
  total_round_num: 400

# Dataset related options
data:
  # Root directory where the data stored
  root: data/
  # Dataset name
  type: cora
  # Use Louvain algorithm to split `Cora`
  splitter: 'louvain'
  # Use fullbatch training, batch_size should be `1`
  batch_size: 1

# Model related options
model:
  # Model type
  type: gcn
  # Hidden dim
  hidden: 64
  # Dropout rate
  dropout: 0.5
  # Number of Class of `Cora`
  out_channels: 7

# Criterion related options
criterion:
  # Criterion type
  type: CrossEntropyLoss

# Trainer related options
trainer:
  # Trainer type
  type: nodefullbatch_trainer

# Train related options
train:
  # Number of local update steps
  local_update_steps: 4
  # Optimizer related options
  optimizer:
    # Learning rate
    lr: 0.25
    # Weight decay
    weight_decay: 0.0005
    # Optimizer type
    type: SGD

# Evaluation related options
eval:
  # Frequency of evaluation
  freq: 1
  # Evaluation metrics, accuracy and number of correct items
  metrics: ['acc', 'correct']
```

### Start with customized functions

FS-G also provides `register` function to set up the FL. Here we provide an example about how to run your own model and data to FS-G.

* Load your data (write in `federatedscope/contrib/data/`):

  ```python
  import copy
  import numpy as np
  
  from torch_geometric.datasets import Planetoid
  from federatedscope.core.splitters.graph import LouvainSplitter
  from federatedscope.register import register_data
  
  
  def my_cora(config=None):
      path = config.data.root
  
      num_split = [232, 542, np.iinfo(np.int64).max]
      dataset = Planetoid(path,
                          'cora',
                          split='random',
                          num_train_per_class=num_split[0],
                          num_val=num_split[1],
                          num_test=num_split[2])
      global_data = copy.deepcopy(dataset)[0]
      dataset = LouvainSplitter(config.federate.client_num)(dataset[0])
  
      data_local_dict = dict()
      for client_idx in range(len(dataset)):
          data_local_dict[client_idx + 1] = dataset[client_idx]
  
      data_local_dict[0] = global_data
      return data_local_dict, config
  
  
  def call_my_data(config):
      if config.data.type == "mycora":
          data, modified_config = my_cora(config)
          return data, modified_config
  
  
  register_data("mycora", call_my_data)
  
  ```

* Build your model (write in `federatedscope/contrib/model/`):

  ```python
  import torch
  import torch.nn.functional as F
  
  from torch.nn import ModuleList
  from torch_geometric.data import Data
  from torch_geometric.nn import GCNConv
  from federatedscope.register import register_model
  
  
  class MyGCN(torch.nn.Module):
      def __init__(self,
                   in_channels,
                   out_channels,
                   hidden=64,
                   max_depth=2,
                   dropout=.0):
          super(MyGCN, self).__init__()
          self.convs = ModuleList()
          for i in range(max_depth):
              if i == 0:
                  self.convs.append(GCNConv(in_channels, hidden))
              elif (i + 1) == max_depth:
                  self.convs.append(GCNConv(hidden, out_channels))
              else:
                  self.convs.append(GCNConv(hidden, hidden))
          self.dropout = dropout
  
      def forward(self, data):
          if isinstance(data, Data):
              x, edge_index = data.x, data.edge_index
          elif isinstance(data, tuple):
              x, edge_index = data
          else:
              raise TypeError('Unsupported data type!')
  
          for i, conv in enumerate(self.convs):
              x = conv(x, edge_index)
              if (i + 1) == len(self.convs):
                  break
              x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
          return x
  
  
  def gcnbuilder(model_config, input_shape):
      x_shape, num_label, num_edge_features = input_shape
      model = MyGCN(x_shape[-1],
                    model_config.out_channels,
                    hidden=model_config.hidden,
                    max_depth=model_config.layer,
                    dropout=model_config.dropout)
      return model
  
  
  def call_my_net(model_config, local_data):
      # Please name your gnn model with prefix 'gnn_'
      if model_config.type == "gnn_mygcn":
          model = gcnbuilder(model_config, local_data)
          return model
  
  
  register_model("gnn_mygcn", call_my_net)
  
  ```

- Run with following command to start:

  ```bash
  python federatedscope/main.py --cfg federatedscope/gfl/baseline/example.yaml data.type mycora model.type gnn_mygcn
  ```

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

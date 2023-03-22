### Vertical Federated Learning

We provide an example for vertical federated learning, you can run with:
```bash
python3 ../main.py --cfg vertical_fl.yaml
```

You can specify customized configurations in `vertical_fl.yaml`, such as `data.type` and `federate.total_round_num`. 
More details of the provided example can be found in [Tutorial](https://federatedscope.io/docs/cross-silo/).

Note that FederatedScope only provide an `abstract_paillier`, user can refer to [pyphe](https://github.com/data61/python-paillier/blob/master/phe/paillier.py) for the detail implementation, or adopt other homomorphic encryption algorithms.

## Tree-based Models
FederatedScope-Tree is an efficient package for VFL. We provide a hands-on tutorial here. For more details, please refer to our paper ..

### Run with example config
You can run with the following as an example:
```bash
python federatedscope/main.py --cfg federatedscope/vertical_fl/xgb_base/baseline/xgb_base_on_abalone.yaml
```
### Advanced
We provide two typed of tree-based models: feature-gathering tree-based model and label-scattering tree-based model. 

For feature-gathering models, we implement XGBoost, GBDT and random forest. For label-scattering models, we implement XGBoost and GBDT.

#### Basic
You can set the model and algorithm in .yaml files, see the following as an example:
```bash
use_gpu: False
device: 0
backend: torch
federate:
  mode: standalone
  client_num: 2
model:
  type: xgb_tree  # xgb_tree or gbdt_tree or random_forest
  lambda_: 0.1
  gamma: 0
  num_of_trees: 10
  max_tree_depth: 6
data:
  root: data/
  type: abalone
  splits: [0.8, 0.2]
dataloader:
  type: raw
  batch_size: 4177
criterion:
  type: RegressionMSELoss  # CrossEntropyLoss, for binary classification
trainer:
  type: verticaltrainer
vertical:
  use: True
  dims: [6, 10]
  algo: 'xgb' # 'xgb' or 'gbdt' or 'rf'
  data_size_for_debug: 0
eval:
  freq: 3
  best_res_update_round_wise_key: test_loss
  ```
The ```model.type``` and ```vertical.algo``` must correspond to each other, there are three choices: ```[xgb_tree, 'xgb']```,```[gbdt_tree, 'gbdt']``` and ```[random_forest, 'rf']```.

#### Privacy-preserving for feature-gathering model
The above .yaml contains no privacy protecting. For feature-gathering model, we provide two privacy-preserving methods for XGBoost, GBDT and random forest. One is [FederBoost: Private Federated Learning for
GBDT](https://arxiv.org/pdf/2011.02796.pdf), and the other is [OpBoost: A Vertical Federated Tree Boosting Framework Based
on Order-Preserving Desensitization](https://arxiv.org/pdf/2210.01318.pdf). You can add it in .yaml file as:
```bash
  ...
vertical:
  protect_object: 'feature_order'
  protect_method: 'dp'
  protect_args: [{'bucket_num': 50, 'epsilon': 3}]
  # protect_args: [{'bucket_num': 50}]
  ...  
``` 
or 
```bash
  ...
vertical:
  protect_object: 'feature_order'
  protect_method: 'op_boost'
  protect_args: [{'algo': 'global', 'epsilon': 2, 'bucket_num': 50}]
  # protect_args: [{'algo': 'adjust', 'epsilon_prt': 2, 'epsilon_ner': 2, 'partition_num': 50}]
  ...
```
For more details, please see [feture_order_protected_train.py](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/vertical_fl/trainer/feature_order_protected_trainer.py).

#### Privacy-preserving for label-gathering model
For label-scattering model, we provide one privacy-preserving method only for XGBoost and GBDT, see [SecureBoost: A Lossless Federated Learning
Framework](https://arxiv.org/pdf/1901.08755.pdf). You can add it in .yaml file as: 
```bash
  ...
vertical:
  mode: 'label_based'
  protect_object: 'grad_and_hess'
  protect_method: 'he'
  key_size: 512
  protect_args: [ { 'bucket_num': 50 } ]
  ...  
``` 

#### Inference procedure
The default procedure is as stated in [SecureBoost: A Lossless Federated Learning
Framework](https://arxiv.org/pdf/1901.08755.pdf). We have implemented two more ways. One uses PHE as [Fed-EINI: An Efficient and Interpretable Inference Framework for Decision Tree Ensembles in Vertical Federat](https://arxiv.org/pdf/2105.09540.pdf), the other uses secret sharing as [https://arxiv.org/pdf/2105.09540.pdf](https://arxiv.org/pdf/2005.08479.pdf). You can set it here:
```bash
  ...
vertical:
  eval: 'homo' # or 'ss' or None 
  ...
```


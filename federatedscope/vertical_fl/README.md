# Vertical Federated Learning

## Linear Model
We provide an example for training linear model in vertical federated learning:
```bash
python3 ../main.py --cfg linear_model/baseline/vertical_fl.yaml
```

You can specify customized configurations in `vertical_fl.yaml`, such as `data.type` and `federate.total_round_num`. 
More details of the provided example can be found in [Tutorial](https://federatedscope.io/docs/cross-silo/).

Note that FederatedScope only provide an `abstract_paillier`, user can refer to [pyphe](https://github.com/data61/python-paillier/blob/master/phe/paillier.py) for the detail implementation, or adopt other homomorphic encryption algorithms.

## Tree-based Models
FederatedScope-Tree is built for training tree-based models in vertical federated learning, such as XGB, GBDT, RF, etc. 
We provide a hands-on tutorial here. 

We categorize tree-based models in vertical FL into two types, i.e., *feature-gathering tree-based models* and *label-scattering tree-based models*, according to their communication and computation protocols, and provide the corresponding implementations.
- Feature-gathering tree-based models: The data parties (i.e., the participants who don't hold labels) send *the orders of their feature values* to the task party (i.e., the participant holds labels) for training, and then the task party calculate and compare the information gains for determining the split rules at the nodes. 
- Label-scattering tree-based models: The task party sends the label-related information (such as gradient and hessian values) to the data parties for calculating the information gains, and all the gains would be retuned to the task parties for compression.


### Configurations
You can set the model and algorithm in yaml files as below:
```bash
use_gpu: False  # Whether to use GPU
device: 0  # Deciding which GPU to use
backend: torch

# Federate learning related options
federate:
  mode: standalone  # `standalone` or `distributed`
  client_num: 2  # number of client
model:
  type: xgb_tree  # xgb_tree or gbdt_tree or random_forest
  # related hyperparameters
  lambda_: 0.1
  gamma: 0
  num_of_trees: 10
  max_tree_depth: 6
  
# Dataset related options
data:
  root: data/  # Root directory where the data stored
  type: abalone  # Dataset name
  splits: [0.8, 0.2]  # splits for training and testing
dataloader:
  type: raw  # Personalized DataLoader
  batch_size: 4177
criterion:
  type: RegressionMSELoss  # CrossEntropyLoss, for binary classification

# Trainer related options
trainer:
  # Trainer type
  type: verticaltrainer  
  
# vertical related options
vertical:
  use: True
  dims: [4, 8] # feature split for two clients, one has feature 0~3, 
  # and the other has feature 4~7
  feature_subsample_ratio: 1.0 # default = 1.0 
  # the proportion of the numbers of features used for training per user.
  algo: 'xgb' # 'xgb' or 'gbdt' or 'rf'
  data_size_for_debug: 0  # use a subset for debug in vfl,
  # 0 indicates using the entire dataset (disable debug mode)

# Evaluation related options
eval:
  # Frequency of evaluation
  freq: 3
  best_res_update_round_wise_key: test_loss
```
Users can specify ```model.type``` and ```vertical.algo``` to use different models. For examples,:
-  XGBoost: `model.type = xgb_tree`, and `vertical.algo = xgb`
-  GBDT: `model.type = gbdt_tree`, and `vertical.algo = gbdt`
-  Random Forest: `model.type = random_forest`, and `vertical.algo = rf`

### Privacy protection algorithms 
#### For feature-gathering model
For feature-gathering models, we provide two kinds of privacy protection algorithms to protect the order of feature values.

One of protection methods is differential privacy (DP). Users can add the following configurations:
```
vertical:
  protect_object: 'feature_order'
  protect_method: 'dp'
  protect_args: [{'bucket_num': 50, 'epsilon': 3}]
  # protect_args: [{'bucket_num': 50}] 
```
`'bucket_num': b` means that we partition the order into $b$ buckets evenly, and each sample in the bucket will stay inside its own bucket with a fixed probability $p$, and with probability $1-p$ it will shuffle to another bucket uniformly and independently, where $p=\frac{e^{\epsilon}}{e^{\epsilon}+b-1}$ and $b$ are parameters to control the strength of privacy protection. 
From the formulation, it can be seen that, when $b$ and $\epsilon$ are small, the strength for privacy protection is strong but the model utility can be affected. When $\epsilon$ is set to `None` (i.e., $p=1$), we just random shuffle the instances within each bucket.

Another protection method is 'op_boost' (global/local) as follows: 

```
vertical:
  protect_object: 'feature_order'
  protect_method: 'op_boost'
  protect_args: [{'algo': 'global', 'lower_bound': lb, 'upper_bound': ub, 'epsilon': 2}]
  # protect_args: [{'algo': 'adjust', 'lower_bound': lb, 'upper_bound': ub, 'epsilon_prt': 2, 'epsilon_ner': 2, 'partition_num': pb}]
```
- `global` means we map the data into the integers between $[lb, ub]$ by affine transformation. For each mapped value $x$, it will be re-mapped to $i\in[lb, ub]$ with probability $$p=\frac{e^{-|x-i|\cdot\epsilon/2}}{\sum_{j\in[lb, ub]} e^{-|x-j|\cdot\epsilon/2}}$$ randomly.

- `adjusting` means we map the data into the integers between $[lb, ub]$, and then partition $[lb, ub]$ into $pb$ buckets evenly. For a value $x$ inside the $m$-th bucket, we first randomly select a bucket $i$ with probability $$p=\frac{e^{-|m-i|\cdot\epsilon_{p\ r\ t}/2}}{\sum_{j\in[lb, ub]}e^{-|m-j|\cdot\epsilon_{p\ r\ t}/2}},$$ then we randomly select a value $v$ in the selected bucket with probability $$p=\frac{e^{-|x-v|\cdot\epsilon_{n\ e\ r}/2}}{\sum_{j\in[lb, ub]} e^{-|x-j|\cdot\epsilon_{n\ e\ r}/2}}.$$ 
When $lb$ and $ub$ are close, $\epsilon, \epsilon_{prt}, \epsilon_{ner}$ and $pb$ are small, the strength for privacy protection is strong but the model utility can be affected.

In `protect_args`, you can also add `bucket_num` to accelerate the training which is similar to the hist algorithm in XGBoost.

The above two protection methods were proposed in  "FederBoost: Private Federated Learning for
GBDT" and "OpBoost: A Vertical Federated Tree Boosting Framework Based on Order-Preserving Desensitization". 

#### For label-scattering model
For label-scattering model, we provide privacy protection algorithms proposed by "SecureBoost: A Lossless Federated Learning Framework".  Users can add the following configurations: 

```
vertical:
  mode: 'label_based'
  protect_object: 'grad_and_hess'
  protect_method: 'he'
  key_size: ks
  protect_args: [ { 'bucket_num': b } ]
```

Specifically, the task party encrypts the label-related information (such as grad and hess for XGBoost , grad and indicator vector for GBDT), and send them to data party. Each data party sort the encrypted information by the order of feature values, and partition them into $b$ buckets evenly, and calculates the partial sums and sends them back to task party for computing best gain.


### Inference procedure

In Inference procedure, we also provide different manners. Users can specify `vertical.eval` to apply secret sharing (`ss`), homomorphic encryption (`he`), or choose to no apply protection method (`''`).

`vertical.eval: ''` means the basic procedure, that is, for each tree, when task party performs inference, for each internal node, he will check the owner of the split feature, and sends a single to the owner. The owner compares the test data and the split value to get the indicator vectors for left and right children, then sends them to task party. Task party continues testing for the next node until the leaf nodes are reached.

`vertical.eval: 'ss'` (coming soon!) means for each tree, task party first SS the weight of each leaf node. And during inference, the indicator vectors of left and right children are also secret shared. By SS multiplication, at the end, each party will get a secret shared piece of the testing results. Then task party receives the pieces of data parties to reveal the exact result. The main advantage is that the indicator vectors are masked.  This is adapted from "Large-Scale Secure XGB for Vertical Federated Learning".

`vertical.eval: 'he'` means for each tree, each party locally get the leaf vector. Here, a leaf vector is a $0-1$ vector of length equals to the number of leaf node plus 1, where $0$ means that the sample must not in this leaf node and $1$ otherwise. Task party put the weight of the leaf node into the leaf vector where the corresponding coordinate has a $1$ and encrypts it by PHE and sends it to one data party. The data party performs dot production between his own leaf vector with encrypted vector, and sends it to the next data party. The next data party dose exactly the same thing. Finally, the last data party sums up each component of the vector and sends it to task party. Task party decrypts it to get the testing result. This method was proposed in "Fed-EINI: An Efficient and Interpretable Inference Framework for Decision Tree Ensembles in Vertical Federated Learning".

### Examples
Several examples are provided in `federatedscope/tree_based_models/baseline/`. The following configurations (provided in `xgb_feature_gathering_dp_on_adult.yaml`) shows how to run XGBoost n Adult dataset, using feature-gathering tree-based model and applying differential privacy for protection.
```shell script
use_gpu: False
device: 0
backend: torch
federate:
  mode: standalone
  client_num: 2
model:
  type: xgb_tree
  lambda_: 0.1
  gamma: 0
  num_of_trees: 10
  max_tree_depth: 3
data:
  root: data/
  type: adult
  splits: [1.0, 0.0]
dataloader:
  type: raw
  batch_size: 2000
criterion:
  type: CrossEntropyLoss
trainer:
  type: verticaltrainer
train:
  optimizer:
    # learning rate for xgb model
    eta: 0.5
vertical:
  use: True
  dims: [7, 14]
  algo: 'xgb'
  protect_object: 'feature_order'
  protect_method: 'dp'
  protect_args: [{'bucket_num': 100, 'epsilon':10}]
  data_size_for_debug: 2000
eval:
  freq: 3
  best_res_update_round_wise_key: test_loss
```

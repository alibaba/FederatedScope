### Vertical Federated Learning

We provide an example for vertical federated learning, you can run with:
```bash
python3 ../main.py --cfg vertical_fl.yaml
```

You can specify customized configurations in `vertical_fl.yaml`, such as `data.type` and `federate.total_round_num`. 
More details of the provided example can be found in [Tutorial](https://federatedscope.io/docs/cross-silo/).

Note that FederatedScope only provide an `abstract_paillier`, user can refer to [pyphe](https://github.com/data61/python-paillier/blob/master/phe/paillier.py) for the detail implementation, or adopt other homomorphic encryption algorithms.

## Tree-based Models
FederatedScope-Tree is an efficient package for VFL. We provide a hands-on tutorial here. 
<!-- For more details, please refer to our paper -->

### Run with example config
You can run with the following as an example:
```bash
python federatedscope/main.py --cfg federatedscope/vertical_fl/xgb_base/baseline/xgb_base_on_abalone.yaml
```
### Advanced
We provide two types of tree-based models: feature-gathering tree-based model and label-scattering tree-based model. In plain-text case, rough speaking, the first model means data parties (parties do not hold the label) send the orders of the values of their features to task party (label owner) for training, while the second model means task party sends his label (or grad, hess) to data party, and receives the partial sums from data parties for training.

For feature-gathering model, we implement XGBoost, GBDT and random forest. For label-scattering models, we implement XGBoost and GBDT.

#### Basic
You can set the model and algorithm in .yaml files, see the following as an example:
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
The ```model.type``` and ```vertical.algo``` must correspond to each other, there are three choices: ```[xgb_tree, 'xgb']```,```[gbdt_tree, 'gbdt']``` and ```[random_forest, 'rf']```.

#### Privacy-preserving for feature-gathering model
The above .yaml contains no privacy protection. For feature-gathering model, we provide two privacy-preserving methods to protect the order of feature values.

The first protection method is use bucket and DP, which can be set in .yaml file as follows:
```bash
  ...
vertical:
  ...
  protect_object: 'feature_order'
  protect_method: 'dp'
  protect_args: [{'bucket_num': 50, 'epsilon': 3}]
  # protect_args: [{'bucket_num': 50}] 
  ...  
```
```'bucket_num': b``` means that we partition the order into $b$ buckets evenly, and each sample in the bucket will stay inside its own bucket with a fixed probability $p$, and with probability $1-p$ it will fly to another bucket uniformly and independently, where $p=\frac{e^{\epsilon}}{e^{\epsilon}+b-1}$ and $b$ are parameters to control the strength for privacy preserving. From the formulation, it can be seen that, when $b$ and $\epsilon$ are smaller, the strength for privacy preserving will be stronger, but the results will become worse. The default value of $b$ is $100$ and the default value of $\epsilon$ is ```None``` which means $p=1$, i.e., we just random shuffle the order inside each bucket.

The second protection method is using 'op_boost' as follows: 

```bash
  ...
vertical:
  ...
  protect_object: 'feature_order'
  protect_method: 'op_boost'
  protect_args: [{'algo': 'global', 'lower_bound': lb, 'upper_bound': ub, 'epsilon': 2}]
  # protect_args: [{'algo': 'adjust', 'lower_bound': lb, 'upper_bound': ub, 'epsilon_prt': 2, 'epsilon_ner': 2, 'partition_num': pb}]
  ...
```
We provide two algorithms: 'global' and 'adjusting '.

```'global'``` means we map the data into the integers between $[lb, ub]$ by affine transformation where $lb<ub$ are integers, and for each mapped value $x$, it will be re-mapped to $i\in[lb, ub]$ with probability $p=\frac{e^{-|x-i|\cdot\epsilon/2}}{\sum_{j\in[lb, ub]e^{-|x-j|\cdot\epsilon/2}}}$ randomly. Finally, we use the order of the values for training. The default setting is ```protect_args: [{'algo': 'global', 'lower_bound': 1, 'upper_bound': 100, 'epsilon': 2}]```. 

```'adjusting'``` means we map the data into the integers between $[lb, ub]$, and then partition  $[lb, ub]$ into $pb$ buckets evenly. For a value $x$ inside the $m$-th bucket, we first randomly select a bucket $i$ with probability $p=\frac{e^{-|m-i|\cdot\epsilon_{prt}/2}}{\sum_{j\in[lb, ub]e^{-|m-j|\cdot\epsilon_{prt}/2}}}$, then we randomly select a value $v$ in the selected bucket with probability  $p=\frac{e^{-|x-v|\cdot\epsilon_{ner}/2}}{\sum_{j\in[lb, ub]e^{-|x-j|\cdot\epsilon_{ner}/2}}}$. Finally, we use the order of the values for training. The default setting is ```protect_args: [{'algo': 'adjusting', 'lower_bound': 1, 'upper_bound': 100, 'epsilon_prt': 2, 'epsilon_ner': 2, 'partition_num': 10}]```. Thus, when $lb$ and $ub$ are closer, $\epsilon, \epsilon_{prt}, \epsilon_{ner}$ and $pb$ are smaller, the strength for privacy preserving will be stronger, but the results will become worse. 

In ```protect_args```, you can also add ```'bucket_num': b``` to accelerate the training which is similar to the hist algorithm in XGBoost.

 For more details, please see [feture_order_protected_train.py](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/vertical_fl/trainer/feature_order_protected_trainer.py).

##### Remark

The above two protection method are proposed in  "FederBoost: Private Federated Learning for
GBDT" and "OpBoost: A Vertical Federated Tree Boosting Framework Based on Order-Preserving Desensitization". 

#### Privacy-preserving for label-scattering model
For label-scattering model, we provide one privacy-preserving method only for XGBoost and GBDT, which is from  "SecureBoost: A Lossless Federated Learning Framework".  The .yaml file is as follows: 

```bash
  ...
vertical:
  ...
  mode: 'label_based'
  protect_object: 'grad_and_hess'
  protect_method: 'he'
  key_size: ks
  protect_args: [ { 'bucket_num': b } ]
  ...  
```

The detail is that task party encrypts the label-related information (such as grad and hess for XGBoost , grad and indicator vector for GBDT), and send them to data party. Each data party sort the encrypted information by the order of feature values, and partition them into $b$ buckets evenly, and calculates the partial sums and sends them back to task party for computing best gain.

The default setting are ```key_size: 3072``` and ```protect_args: [ { 'bucket_num': 100 } ]```

#### Inference procedure

In Inference procedure, we also provide different manners. Users can set in the following:

```
	...
vertical:
	...
	eval: 'homo' # defalut value: ''
	...
```

```vertical.eval: ''``` means the basic procedure, that is, for each tree, when task party do the inference, for each internal node, he will check the owner of the split feature, and sends a single to the owner. The owner compares the test data and the split value to get the indicator vectors for left and right children, then sends them to task party. Task party continues testing for the next node until the leaf nodes are reached.

```vertical.eval: 'SS'``` (coming soon!) means for each tree, task party first SS the weight of each leaf node. And during inference, the indicator vectors of left and right children are also secret shared. By SS multiplication, at the end, each party will get a secret shared piece of the testing results. Then task party receives the pieces of data parties to reveal the exact result. The main advantage is that the indicator vectors are masked. 

```vertical.eval: 'homo'``` means for each tree, each party locally get the leaf vector. Here, a leaf vector is a $0$-$1$ vector of length equals to the number of leaf node plus 1, where $0$ means that the sample must not in this leaf node and $1$ otherwise. Task party put the weight of the leaf node into the leaf vector where the corresponding coordinate has a $1$ and encrypts it by PHE and sends it to one data party. The data party  do the dot production between his own leaf vector with encrypted vector, and sends it to the next data party. The next data party dose exactly the same thing. Finally, the last data party sums up each component of the vector and sends it to task party. Task party decrypts it to get the testing result.


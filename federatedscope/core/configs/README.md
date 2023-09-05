## Configurations
We summarize all the customizable configurations:
- [config.py](#config)
- [cfg_data.py](#data)
- [cfg_model.py](#model)
- [cfg_fl_algo.py](#federated-algorithms)
- [cfg_training.py](#federated-training)
- [cfg_fl_setting.py](#fl-setting)
- [cfg_evaluation.py](#evaluation)
- [cfg_asyn.py](#asynchronous-training-strategies)
- [cfg_differential_privacy.py](#differential-privacy)
- [cfg_hpo.py](#auto-tuning-components)
- [cfg_attack.py](#attack)
- [cfg_llm.py](#llm)

### config
The configurations related to environment of running experiment.

| Name                   | (Type) Default Value | Description                                                  | Note |
| ---------------------- | -------------------- | ------------------------------------------------------------ | ---- |
| `backend`              | (string) 'torch'     | The backend for local training                               | -    |
| `use_gpu`              | (bool) False         | Whether to use GPU                                           | -    |
| `check_completeness`   | (bool) False         | Whether to check the completeness of msg_handler             | -    |
| `verbose`              | (int) 1              | Whether to print verbose logging info                        | -    |
| `print_decimal_digits` | (int) 6              | How many decimal places we print out using logger            | -    |
| `device`               | (int) -1             | Specify the device for training                              | -    |
| `seed`                 | (int) 0              | Random seed                                                  | -    |
| `outdir`               | (string) ''          | The dir used to save log, exp_config, models, etc,.          | -    |
| `expname`              | (string) ''          | Detailed exp name to distinguish different sub-exp           | -    |
| `expname_tag`          | (string) ''          | Detailed exp tag to distinguish different sub-exp with the same expname | -    |


### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

|                     Name                     |  (Type) Default Value | Description | Note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:--------------------------------------------:|:-----:|:---------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                 `data.root`                  | (string) 'data' | The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|                 `data.type`                  | (string) 'toy' | Dataset name | CV: 'femnist', 'celeba' ; NLP: 'shakespeare', 'subreddit', 'twitter'; Graph: 'cora', 'citeseer', 'pubmed', 'dblp_conf', 'dblp_org', 'csbm', 'epinions', 'ciao', 'fb15k-237', 'wn18', 'fb15k' , 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI', 'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP', 'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX', 'graph_multi_domain_mol', 'graph_multi_domain_small', 'graph_multi_domain_mix', 'graph_multi_domain_biochem'; MF: 'vflmovielens1m', 'vflmovielens10m', 'hflmovielens1m', 'hflmovielens10m', 'vflnetflix', 'hflnetflix'; Tabular: 'toy', 'synthetic'; External dataset: 'DNAME@torchvision', 'DNAME@torchtext', 'DNAME@huggingface_datasets', 'DNAME@openml'. |
|               `data.file_path`               | (string) '' | The path to the data file, only makes effect when data.type = 'file' | - |
|                 `data.args`                  | (list) [] | Args for the external dataset | Used for external dataset, eg. `[{'download': False}]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|               `data.save_data`               | (bool) False | Whether to save the generated toy data | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               `data.splitter`                | (string) '' | Splitter name for standalone dataset | Generic splitter: 'lda'; Graph splitter: 'louvain', 'random', 'rel_type', 'graph_type', 'scaffold', 'scaffold_lda', 'rand_chunk'                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|             `data.splitter_args`             | (list) [] | Args for splitter. | Used for splitter, eg. `[{'alpha': 0.5}]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|               `data.transform`               | (list) [] | Transform for x of data | Used in `get_item` in torch.dataset, eg. `[['ToTensor'], ['Normalize', {'mean': [0.9637], 'std': [0.1592]}]]`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
|           `data.target_transform`            | (list) [] | Transform for y of data | Use as `data.transform`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|             `data.pre_transform`             | (list) [] | Pre_transform for `torch_geometric` dataset | Use as `data.transform`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|           `dataloader.batch_size`            | (int) 64 | batch_size for DataLoader | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|            `dataloader.drop_last`            | (bool) False | Whether drop last batch (if the number of last batch is smaller than batch_size) in DataLoader | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|              `dataloader.sizes`              | (list) [10, 5] | Sample size for graph DataLoader | The length of `dataloader.sizes` must meet the layer of GNN models.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|             `dataloader.shuffle`             | (bool) True | Shuffle train DataLoader | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|           `data.server_holds_all`            | (bool) False | Only use in global mode, whether the server (workers with idx 0) holds all data, useful in global training/evaluation case | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               `data.subsample`               | (float) 1.0 |  Only used in LEAF datasets, subsample clients from all clients | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|                `data.splits`                 | (list) [0.8, 0.1, 0.1] | Train, valid, test splits | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `data.` </br>`consistent_label_distribution` | (bool) True | Make label distribution of train/val/test set over clients keep consistent during splitting | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|               `data.cSBM_phi`                | (list) [0.5, 0.5, 0.5] | Phi for cSBM graph dataset | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|                `data.loader`                 | (string) '' | Graph sample name, used in minibatch trainer | 'graphsaint-rw': use `GraphSAINTRandomWalkSampler` as DataLoader; 'neighbor': use `NeighborSampler` as DataLoader.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|           `dataloader.num_workers`           | (int) 0 | num_workers in DataLoader | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|           `dataloader.walk_length`           | (int) 2 | The length of each random walk in graphsaint. | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|            `dataloader.num_steps`            | (int) 30 | The number of iterations per epoch in graphsaint. | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|             `data.quadratic.dim`             | (int) 1 | Dim of synthetic quadratic  dataset | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|          `data.quadratic.min_curv`           | (float) 0.02 | Min_curve of synthetic quadratic  dataset | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|          `data.quadratic.max_curv`           | (float) 12.5 | Max_cur of synthetic quadratic  dataset | -                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |


### Model

The configurations related to the model are defined in `cfg_model.py`.  
| [General](#model-general) | [Criterion](#criterion) | [Regularization](#regularizer) | 

#### Model-General
|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `model.`</br> `model_num_per_trainer` |     (int) 1     | Number of model per trainer |                                                                 some methods may leverage more                                                                 |
| `model.type` | (string) 'lr' | The model name used in FL | CV: 'convnet2', 'convnet5', 'vgg11', 'lr'; NLP: 'LSTM', 'MODEL@transformers'; Graph:  'gcn', 'sage', 'gpr', 'gat', 'gin', 'mpnn';  Tabular: 'mlp', 'lr', 'quadratic'; MF: 'vmfnet', 'hmfnet' |
| `model.use_bias` | (bool) True | Whether use bias in lr model | - |
| `model.task` | (string) 'node' | The task type of model, the default is `Classification` | NLP: 'PreTraining', 'QuestionAnswering', 'SequenceClassification', 'TokenClassification', 'Auto', 'WithLMHead'; Graph: 'NodeClassification', 'NodeRegression', 'LinkClassification', 'LinkRegression', 'GraphClassification', 'GraphRegression', |
| `model.hidden` | (int) 256 | Hidden layer dimension | - |
| `model.dropout` | (float) 0.5 | Dropout ratio | - |
| `model.in_channels` | (int) 0 | Input channels dimension | If 0, model will be built by `data.shape` |
| `model.out_channels` | (int) 1 | Output channels dimension | - |
| `model.layer` | (int) 2 | Model layer | - |
| `model.graph_pooling` | (string) 'mean' | Graph pooling method in graph-level task | 'add', 'mean' or 'max' |
| `model.embed_size` | (int) 8 | `embed_size` in LSTM | - |
| `model.num_item` | (int) 0 | Number of items in MF. | It will be overwritten by the real value of the dataset. |
| `model.num_user` | (int) 0 | Number of users in MF. | It will be overwritten by the real value of the dataset. |

#### Criterion

|            Name            | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `criterion.type` |     (string) 'MSELoss'     | Criterion type |                                                                                           Chosen from https://pytorch.org/docs/stable/nn.html#loss-functions , eg. 'CrossEntropyLoss', 'L1Loss', etc.                                                                                            |

#### Regularizer

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `regularizer.type` |     (string) ' '     | The type of the regularizer |                                                                 Chosen from [`proximal_regularizer`]                                                                 |
| `regularizer.mu` | (float) 0 | The factor that controls the loss of the regularization term | - |


### Federated Algorithms 
The configurations related to specific federated algorithms, which are defined in `cfg_fl_algo.py`.

| [FedOPT](#fedopt-for-fedopt-algorithm) | [FedProx](#fedprox-for-fedprox-algorithm) | [personalization](#personalization-for-personalization-algorithms) | [fedsageplus](#fedsageplus-for-fedsageplus-algorithm) | [gcflplus](#gcflplus-for-gcflplus-algorithm) | [flitplus](#flitplus-for-flitplus-algorithm) |

#### `fedopt`: for FedOpt algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedopt.use` | (bool) False | Whether to run FL courses with FedOpt algorithm. | If False, all the related configurations (cfg.fedopt.xxx) would not take effect. |
| `fedopt.optimizer.type` | (string) 'SGD' | The type of optimizer used for FedOpt algorithm. | Currently we support all optimizers build in PyTorch (The modules under torch.optim). |
| `fedopt.optimizer.lr` | (float) 0.1 | The learning rate used in for FedOpt optimizer. | - |
#### `fedprox`: for FedProx algorithm 
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedprox.use` | (bool) False | Whether to run FL courses with FedProx algorithm. | If False, all the related configurations (cfg.fedprox.xxx) would not take effect. |
| `fedprox.mu` | (float) 0.0 | The hyper-parameter $\mu$ used in FedProx algorithm. | - |
#### `personalization`: for personalization algorithms
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `personalization.local_param` | (list of str) [] | The client-distinct local param names, e.g., ['pre', 'bn'] | - |
| `personalization.`</br> `share_non_trainable_para` | (bool) False | Whether transmit non-trainable parameters between FL participants | - |
| `personalization.`</br> `local_update_steps` | (int) -1 | The local training steps for personalized models | By default, -1 indicates that the local model steps will be set to be the same as the valid `train.local_update_steps` |
| `personalization.regular_weight` | (float) 0.1 | The regularization factor used for model para regularization methods such as Ditto and pFedMe. | The smaller the regular_weight is, the stronger emphasising on personalized model. |
| `personalization.lr` | (float) 0.0 | The personalized learning rate used in personalized FL algorithms. | The default value 0.0 indicates that the value will be set to be the same as `train.optimizer.lr` in case of users have not specify a valid `personalization.lr` |
| `personalization.K` | (int) 5 | The local approximation steps for pFedMe. | - |
| `personalization.beta` | (float) 5 | The average moving parameter for pFedMe. | - |
#### `fedsageplus`: for fedsageplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `fedsageplus.num_pred` | (int) 5 | Number of nodes generated by the generator | - |
| `fedsageplus.gen_hidden` | (int) 128 | Hidden layer dimension of generator | - |
| `fedsageplus.hide_portion` | (float) 0.5 | Hide graph portion | - |
| `fedsageplus.fedgen_epoch` | (int) 200 | Federated training round for generator | - |
| `fedsageplus.loc_epoch` | (int) 1 | Local pre-train round for generator | - |
| `fedsageplus.a` | (float) 1.0 | Coefficient for criterion number of missing node | - |
| `fedsageplus.b` | (float) 1.0 | Coefficient for criterion feature | - |
| `fedsageplus.c` | (float) 1.0 | Coefficient for criterion classification | - |
#### `gcflplus`: for gcflplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `gcflplus.EPS_1` | (float) 0.05 | Bound for mean_norm | - |
| `gcflplus.EPS_2` | (float) 0.1 | Bound for max_norm | - |
| `gcflplus.seq_length` | (int) 5 | Length of the gradient sequence | - |
| `gcflplus.standardize` | (bool) False | Whether standardized dtw_distances | - |
#### `flitplus`: for flitplus algorithm
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `flitplus.tmpFed` | (float) 0.5 |  gamma in focal loss (Eq.4) | - |
| `flitplus.lambdavat` | (float) 0.5 | lambda in phi (Eq.10) | - |
| `flitplus.factor_ema` | (float) 0.8 | beta in omega (Eq.12) | - |
| `flitplus.weightReg` | (float) 1.0 | balance lossLocalLabel and lossLocalVAT | - |


### Federated training
The configurations related to federated training are defined in `cfg_training.py`.
Considering it's infeasible to list all the potential arguments for optimizers and schedulers, we allow the users to add new parameters directly under the corresponding namespace. 
For example, we haven't defined the argument `train.optimizer.weight_decay` in `cfg_training.py`, but the users are allowed directly use it. 
If the optimizer doesn't require the argument named `weight_decay`, an error will be raised. 

| [Local Training](#local-training) | [Finetune](#fine-tuning) | [Grad Clipping](#grad-clipping) | [Early Stop](#early-stop) | 

#### Local training
The following configurations are related to the local training. 

|            Name            | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                         |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `train.local_update_steps` |       (int) 1        |       The number of local training steps.        |                                                                                          -                                                                                           |
|   `train.batch_or_epoch`   |   (string) 'batch'   |           The type of local training.            |               `train.batch_or_epoch` specifies the unit that `train.local_update_steps` adopts. All new parameters will be used as arguments for the chosen optimizer.               |
|     `train.optimizer`      |          -           |                        -                         |                     You can add new parameters under `train.optimizer` according to the optimizer, e.g., you can set momentum by `cfg.train.optimizer.momentum`.                     |
|   `train.optimizer.type`   |    (string) 'SGD'    |  The type of optimizer used in local training.   |                                               Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                |
|    `train.optimizer.lr`    |     (float) 0.1      |  The learning rate used in the local training.   |                                                                                          -                                                                                           |
|     `train.scheduler`      |          -           |                        -                         | Similar with `train.optimizer`, you can add new parameters as you need, e.g., `train.scheduler.step_size=10`. All new parameters will be used as arguments for the chosen scheduler. |
|   `train.scheduler.type`   |     (string) ''      | The type of the scheduler used in local training |                                         Currently we support all schedulers build in PyTorch (The modules under `torch.optim.lr_scheduler`).                                         |
|   `train.is_enable_half`   |     (bool) False     |            Whether use half precision            |                                                             When model is too large, users can use half-precision model                                                              |

#### Fine tuning
The following configurations are related to the fine tuning.

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `finetune.before_eval` |     (bool) False     |      Indicator of fintune before evaluation      | If `True`, the clients will fine tune its model before each evaluation. Note the fine tuning is only conducted before evaluation and won't influence the upload weights in each round. |
| `finetune.local_update_steps` |       (int) 1        |       The number of local fine tune steps        |                                                                                           -                                                                                            |
| `finetune.batch_or_epoch` |   (string) `batch`   |          The type of local fine tuning.          |                                   Similar with `train.batch_or_epoch`, `finetune.batch_or_epoch` specifies the unit of `finetune.local_update_steps`                                   |
| `finetune.optimizer` |          -           |                        -                         |            You can add new parameters under `finetune.optimizer` according to the type of optimizer. All new parameters will be used as arguments for the chosen optimizer.            |
| `finetune.optimizer.type` |    (string) 'SGD'    |  The type of the optimizer used in fine tuning.  |                                                Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                 |
| `finetune.optimizer.lr` |     (float) 0.1      |   The learning rate used in local fine tuning    |                                                                                           -                                                                                            |
| `finetune.scheduler` |          -           | - |                   Similar with `train.scheduler`, you can add new parameters as you need, and all new parameters will be used as arguments for the chosen scheduler.                   |

#### Grad Clipping
The following configurations are related to the grad clipping.  

|            Name            | (Type) Default Value |                   Description                    |                                                                                          Note                                                                                          |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `grad.grad_clip` |     (float) -1.0     | The threshold used in gradient clipping. |                                                                 `grad.grad_clip < 0` means we don't clip the gradient.                                                                 |

#### Early Stop

|                   Name                   | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                          |
|:----------------------------------------:|:--------------------:|:------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|          `early_stop.patience`           | (int) 5 |  How long to wait after last time the monitored metric improved. |                        Note that the actual_checking_round = `early_step.patience` * `eval.freq`. To disable the early stop, set the `early_stop.patience` <=0                        |
|            `early_stop.delta`            | (float) 0. |  Minimum change in the monitored metric to indicate a improvement. |                                                                                           -                                                                                           |
|   `early_stop.improve_indicaator_mode`   | (string) 'best' | Early stop when there is no improvement within the last `early_step.patience` rounds, in ['mean', 'best'] |                                                                             Chosen from 'mean' or 'best'                                                                              |


### FL Setting
The configurations related to FL settings are defined in `cfg_fl_setting.py`.

| [General](#federate-general-fl-setting) | [Distribute](#distribute-for-distribute-mode) | [Vertical](#vertical-for-vertical-federated-learning) | 

#### `federate`: general fl setting
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `federate.client_num` | (int) 0 | The number of clients that involves in the FL courses. | It can set to 0 to automatically specify by the partition of dataset. |
| `federate.sample_client_num` | (int) -1 | The number of sampled clients in each training round. | - |
| `federate.sample_client_rate` | (float) -1.0 | The ratio of sampled clients in each training round. | - |
| `federate.unseen_clients_rate` | (float) 0.0 | The ratio of clients served as unseen clients, which would not be used for training and only for evaluation. | - |
| `federate.total_round_num` | (int) 50 | The maximum training round number of the FL course. | - |
| `federate.mode` | (string) 'standalone' </br> Choices: {'standalone', 'distributed'} | The running mode of the FL course. | - |
| `federate.share_local_model` | (bool) False | If `True`, only one model object is created in the FL course and shared among clients for efficient simulation. | - | 
| `federate.data_weighted_aggr` | (bool) False | If `True`, the weight of aggregator is the number of training samples in dataset. | - |
| `federate.online_aggr` | (bool) False | If `True`, an online aggregation mechanism would be applied for efficient simulation. | - | 
| `federate.make_global_eval` | (bool) False | If `True`, the evaluation would be performed on the server's test data, otherwise each client would perform evaluation on local test set and the results would be merged. | - |
| `federate.use_diff` | (bool) False | If `True`, the clients would return the variation in local training (i.e., $\delta$) instead of the updated models to the server for federated aggregation. | - | 
| `federate.merge_test_data` | (bool) False | If `True`, clients' test data would be merged and perform global evaluation for efficient simulation. | - |
| `federate.method` | (string) 'FedAvg' | The method used for federated aggregation. | We support existing federated aggregation algorithms (such as 'FedAvg/FedOpt'), 'global' (centralized training), 'local' (isolated training), personalized algorithms ('Ditto/pFedMe/FedEM'), and allow developer to customize. | 
| `federate.ignore_weight` | (bool) False | If `True`, the model updates would be averaged in federated aggregation. | - |
| `federate.use_ss` | (bool) False | If `True`, additively secret sharing would be applied in the FL course. | Only used in vanilla FedAvg in this version. | 
| `federate.restore_from` | (string) '' | The checkpoint file to restore the model. | - |
| `federate.save_to` | (string) '' | The path to save the model. | - | 
| `federate.join_in_info` | (list of string) [] | The information requirements (from server) for joining in the FL course. | We support 'num_sample/client_resource' and allow user customization.
| `federate.sampler` | (string) 'uniform' </br> Choices: {'uniform', 'group'} | The sample strategy of server used for client selection in a training round. | - |
| `federate.` </br>`resource_info_file` | (string) '' | the device information file to record computation and communication ability | - | 
| `federate.process_num` | (int) 1 | The number of parallel processes. It only takes effect when `use_gpu=True`, `backend='torch'`, `federate.mode='standalone'` and `federate.share_local_model=False`, and the value is required to be not greater than the number of GPUs. | - |
#### `distribute`: for distribute mode
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `distribute.use` | (bool) False | Whether to run FL courses with distribute mode. | If `False`, all the related configurations (`cfg.distribute.xxx`) would not take effect.  |
| `distribute.server_host` | (string) '0.0.0.0' | The host of server's ip address for communication | - |
| `distribute.server_port` | (string) 50050 | The port of server's ip address for communication | - |
| `distribute.client_host` | (string) '0.0.0.0' | The host of client's ip address for communication | - |
| `distribute.client_port` | (string) 50050 | The port of client's ip address for communication | - |
| `distribute.role` | (string) 'client' </br> Choices: {'server', 'client'} | The role of the worker | - |
| `distribute.data_idx` | (int) -1 | It is used to specify the data index in distributed mode when adopting a centralized dataset for simulation (formatted as {data_idx: data/dataloader}). | `data_idx=-1` means that the entire dataset is owned by the participant. And we randomly sample the index in simulation for other invalid values excepted for -1.
| `distribute.` </br>`grpc_max_send_message_length` | (int) 100 * 1024 * 1024 | The maximum length of sent messages | - |
| `distribute.` </br>`grpc_max_receive_message_length` | (int) 100 * 1024 * 1024 | The maximum length of received messages | - |
| `distribute.grpc_enable_http_proxy` | (bool) False | Whether to enable http proxy | - |
#### `vertical`: for vertical federated learning
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `vertical.use` | (bool) False | Whether to run vertical FL. | If `False`, all the related configurations (`cfg.vertical.xxx`) would not take effect.  |
| `vertical.encryption` | (string) `paillier` | The encryption algorithms used in vertical FL. | - |
| `vertical.dims` | (list of int) [5,10] | The dimensions of the input features for participants. | - |
| `vertical.key_size` | (int) 3072 | The length (bit) of the public keys. | - | 


### Evaluation
The configurations related to monitoring and evaluation, which are adefined in `cfg_evaluation.py`.

| [General](#evaluation-general) | [WandB](#wandb-for-wandb-tracking-and-visualization) |

#### Evaluation General
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `eval.freq` | (int) 1 | The frequency we conduct evaluation. | - |
| `eval.metrics` | (list of str) [] | The names of adopted evaluation metrics. | By default, we calculate the ['loss', 'avg_loss', 'total'], all the supported metric can be find in `core/monitors/metric_calculator.py` |
| `eval.split` | (list of str) ['test', 'val'] | The data splits' names we conduct evaluation. | - |
| `eval.report` | (list of str) ['weighted_avg', 'avg', 'fairness', 'raw'] | The results reported forms to loggers | By default, we report comprehensive results, - `weighted_avg` and `avg` indicate the weighted average and uniform average over all evaluated clients; - `fairness` indicates report fairness-related results such as individual performance and std across all evaluated clients; - `raw` indicates that we save and compress all clients' individual results without summarization, and users can flexibly post-process the saved results further.|
| `eval.`</br> `best_res_update_round_wise_key` | (str) 'val_loss' | The metric name we used to as the primary key to check the performance improvement at each evaluation round. | - |
| `eval.monitoring` | (list of str) [] | Extended monitoring methods or metric, e.g., 'dissim' for B-local dissimilarity | - |
| `eval.count_flops` | (bool) True | Whether to count the flops during the FL courses. | - |
#### `wandb`: for wandb tracking and visualization
| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `wandb.use` | (bool) False | Whether to use wandb to track and visualize the FL dynamics and results. | If `False`, all the related configurations (`wandb.xxx`) would not take effect. |
| `wandb.name_user` | (str) '' | the user name used in wandb management | - |
| `wandb.name_project` | (str) '' | the project name used in wandb management | - |
| `wandb.online_track` | (bool) True | whether to track the results in an online manner, i.e., log results at every evaluation round | - |
| `wandb.client_train_info` | (bool) True | whether to track the training info of clients | - |


### Asynchronous Training Strategies
The configurations related to applying asynchronous training strategies in FL are defined in `cfg_asyn.py`.

| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `asyn.use` | (bool) False | Whether to use asynchronous training strategies. | If `False`, all the related configurations (`cfg.asyn.xxx`) would not take effect.  |
| `asyn.time_budget` | (int/float) 0 | The predefined time budget (seconds) for each training round. | `time_budget`<=0 means the time budget is not applied. |
| `asyn.min_received_num` | (int) 2 | The minimal number of received feedback for the server to trigger federated aggregation. | - |
| `asyn.min_received_rate` | (float) -1.0 | The minimal ratio of received feedback w.r.t. the sampled clients for the server to trigger federated aggregation. | - |
| `asyn.staleness_toleration` | (int) 0 | The threshold of the tolerable staleness in federated aggregation. | - | 
| `asyn.` </br>`staleness_discount_factor` | (float) 1.0 | The discount factor for the staled feedback in federated aggregation. | - |
| `asyn.aggregator` | (string) 'goal_achieved' </br> Choices: {'goal_achieved', 'time_up'} | The condition for federated aggregation. | 'goal_achieved': perform aggregation when the defined number of feedback has been received; 'time_up': perform aggregation when the allocated time budget has been run out. |
| `asyn.broadcast_manner` | (string) 'after_aggregating' </br> Choices: {'after_aggregating', 'after_receiving'} | The broadcasting manner of server. | 'after_aggregating': broadcast the up-to-date global model after performing federated aggregation; 'after_receiving': broadcast the up-to-date global model after receiving the model update from clients. |
| `asyn.overselection` | (bool) False | Whether to use the overselection technique | - |


### Differential Privacy
| [NbAFL](#nbafl) | [SGDMF](#sgdmf) | 

#### NbAFL
The configurations related to NbAFL method. 

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `nbafl.use` |     (bool) False     | The indicator of the NbAFL method.         | - |
| `nbafl.mu` |      (float) 0.      | The argument $\mu$ in NbAFL.               | - | 
| `nbafl.epsilon` |     (float) 100.     | The $\epsilon$-DP guarantee used in NbAFL. | - |
| `nbafl.w_clip` |      (float) 1.      | The threshold used for weight clipping.    | - |
| `nbafl.constant` |     (float) 30. | The constant used in NbAFL.                | - |

#### SGDMF
The configurations related to SGDMF method (only used in matrix factorization tasks).

|        Name        | (Type) Default Value | Description                        | Note                                                    |
|:------------------:|:--------------------:|:-----------------------------------|:--------------------------------------------------------|
|    `sgdmf.use`     |     (bool) False     | The indicator of the SGDMF method. |                                                         |
|     `sgdmf.R`      |      (float) 5.      | The upper bound of rating.         | -                                                       |
|  `sgdmf.epsilon`   |      (float) 4.      | The $\epsilon$ used in DP.         | -                                                       |
|   `sgdmf.delta`    |     (float) 0.5      | The $\delta$ used in DP.           | -                                                       |
|  `sgdmf.constant`  |      (float) 1. | The constant in SGDMF | -                                                       |
| `dagaloader.theta` | (int) -1 | - | -1 means per-rating privacy, otherwise per-user privacy |


### Auto-tuning Components

These arguments are exposed for customizing our provided auto-tuning components.

| [General](#auto-tunning-general) | [SHA](#successive-halving-algorithm-sha) | [FedEx](#fedex) | [Wrappers for FedEx](#wrappers-for-fedex) | 

#### Auto-tunning General

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.working_folder` |     (string) 'hpo'     | Save model checkpoints and search space configurations to this folder.         | Trials in the next stage of an iterative HPO algorithm can restore from the checkpoints of their corresponding last trials. |
| `hpo.ss` |     (string) 'hpo'     | File path of the .yaml that specifying the search space.         | - |
| `hpo.num_workers` |     (int) 0     | The number of threads to concurrently attempt different hyperparameter configurations.         | Multi-threading is banned in current version. |
| `hpo.init_cand_num` |     (int) 16     | The number of initial hyperparameter configurations sampled from the search space.         | - |
| `hpo.larger_better` |     (bool) False     | The indicator of whether the larger metric is better.         | - |
| `hpo.scheduler` |     (string) 'rs' </br> Choices: {'rs', 'sha', 'wrap_sha'}     | Which algorithm to use.         | - |
| `hpo.metric` |     (string) 'client_summarized_weighted_avg.val_loss'     | Metric to be optimized.         | - |

#### Successive Halving Algorithm (SHA)

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.sha.elim_rate` |     (int) 3     | Reserve only top 1/`hpo.sha.elim_rate` hyperparameter configurations in each state.        | - |
| `hpo.sha.budgets` |     (list of int) []     | Budgets for each SHA stage.        | - |


#### FedEx

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.fedex.use` |     (bool) False     | Whether to use FedEx.        | - |
| `hpo.fedex.ss` |     (striing) ''     | Path of the .yaml specifying the search space to be explored.        | - |
| `hpo.fedex.flatten_ss` |     (bool) True     | Whether the search space has been flattened.        | - |
| `hpo.fedex.eta0` |     (float) -1.0     | Initial learning rate.        | -1.0 means automatically determine the learning rate based on the size of search space. |
| `hpo.fedex.sched` |     (string) 'auto' </br> Choices: {'auto', 'adaptive', 'aggressive', 'constant', 'scale' } | The strategy to update step sizes    | - |
| `hpo.fedex.cutoff` |     (float) 0.0 | The entropy level below which to stop updating the config.        | - |
| `hpo.fedex.gamma` |     (float) 0.0 | The discount factor; 0.0 is most recent, 1.0 is mean.        | - |
| `hpo.fedex.diff` |     (bool) False | Whether to use the difference of validation losses before and after the local update as the reward signal.        | - |

#### Wrappers for FedEx 

| Name | (Type) Default Value | Description                                | Note |
|:----:|:--------------------:|:-------------------------------------------|:-----|
| `hpo.table.eps` |     (float) 0.1 | The probability to make local perturbation.        | Larger values lead to drastically different arms of the bandit FedEx attempts to solve. |
| `hpo.table.num` |     (int) 27 | The number of arms of the bandit FedEx attempts to solve.        | - |
| `hpo.table.idx` |     (int) 0 | The key (i.e., name) of the hyperparameter wrapper considers.        | No need to change this argument. |


### Attack 

The configurations related to the data/dataset are defined in `cfg_attack.py`.

| [Privacy Attack](#for-privacy-attack) | [Back-door Attack](#for-back-door-attack) | 


#### For Privacy Attack
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
`attack.attack_method` | (str) '' | Attack method name | Choices: {'gan_attack', 'GradAscent', 'PassivePIA', 'DLG', 'IG', 'backdoor'} |
`attack.target_label_ind` | (int) -1 | The target label to attack | Used in class representative attack (GAN based method) and back-door attack; defult -1 means no label to target|
`attack.attacker_id` | (int) -1 | The id of the attack client | Default -1 means no client as attacker; Used in both privacy attack and back-door attack when client is the attacker |
`attack.reconstruct_lr `| (float) 0.01 | The learning rate of the optimization based training data/label inference attack|-|
`attack.reconstruct_optim` | (str) 'Adam' | The learning rate of the optimization based training data/label inference attack|Choices: {'Adam', 'SGD', 'LBFGS'}|
`attack.info_diff_type` | (str) 'l2' | The distance to compare the ground-truth info (gradients or model updates) and the info generated by the dummy data. | Options: 'l2', 'l1', 'sim' representing L2, L1 and cosin similarity |
`attack.max_ite` | (int) 400 | The maximum iteration of the optimization based training data/label inference attack |-|
`attack.alpha_TV` | (float) 0.001 | The hyperparameter of the total variance term | Used in the mehtod invert gradint |
`attack.inject_round` | (int) 0 | The round to start performing the attack actions |-|
`attack.classifier_PIA` | (str) 'randomforest' | The property inference classifier name |-|
 `attack.mia_simulate_in_round`|(int) 20 | The round to add the target data into training batch| Used When simulate the case that the target data are in the training set|
 `attack. mia_is_simulate_in` | (bool) False | whether simulate the case that the target data are in the training set||

#### For Back-door Attack
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
`attack.edge_path` |(str) 'edge_data/' | The folder where the ood data used by edge-case backdoor attacks located  |-|
`attack.trigger_path` |(str) 'trigger/'|The folder where the trigger pictures used by pixel-wise backdoor attacks located  |-|
`attack.setting` | (str) 'fix'| The setting about how to select the attack client. |Choices:{'fix', 'single', and 'all'}, 'single' setting means the attack client can be only selected in the predefined round (cfg.attack.insert_round). 'all' setting means the attack client can be selected in all round. 'fix' setting means that the attack client can be selected every freq round. freq has beed defined in the cfg.attack.freq keyword.|
`attack.freq` | (int) 10 |This keyword is used in the 'fix' setting. The attack client can be selected every freq round.|-| 
`attack.insert_round` |(int) 100000 |This keyword is used in the 'single' setting. The attack client can be only selected in the insert_round round.|-|
`attack.mean` |(list) [0.9637] |The mean value which is used in the normalization procedure of poisoning data. |Notice: The length of this list must be same as the number of channels of used dataset.|
`attack.std` |(list) [0.1592] |The std value which is used in the normalization procedure of poisoning data.|Notice: The length of this list must be same as the number of channels of used dataset.|
`attack.trigger_type`|(str) 'edge'|This keyword represents the type of used triggers|Choices: {'edge', 'gridTrigger', 'hkTrigger', 'sigTrigger', 'wanetTrigger', 'fourCornerTrigger'}|
`attack.label_type` |(str) 'dirty'| This keyword represents the type of used attack.|It contains 'dirty'-label and 'clean'-label attacks. Now, we only support 'dirty'-label attack. |
`attack.edge_num` |(int) 100 | This keyword represents the number of used good samples for edge-case attack.|-|
`attack.poison_ratio` |(float) 0.5|This keyword represents the percentage of samples with pixel-wise triggers in the local dataset of attack client|-|
`attack.scale_poisoning` |(bool) False| This keyword represents whether to use the model scaling attack for attack client. |-|
`attack.scale_para` |(float) 1.0 |This keyword represents the value to amplify the model update when conducting the model scaling attack.|-|
`attack.pgd_poisoning` |(bool) False|This keyword represents whether to use the pgd to train the local model for attack client. |-|
`attack.pgd_lr` | (float) 0.1 |This keyword represents learning rate of pgd training for attack client.|-|
`attack.pgd_eps`|(int) 2 | This keyword represents perturbation budget of pgd training for attack client.|-|
`attack.self_opt` |(bool) False |This keyword represents whether to use his own training procedure for attack client.|-|
`attack.self_lr` |(float) 0.05|This keyword represents learning rate of his own training procedure for attack client.|-|
`attack.self_epoch` |(int) 6 |This keyword represents epoch number of his own training procedure for attack client.|-|

### LLM
The configurations related to LLMs are defined in `cfg_llm.py`.

| [General](#llm-general) | [Inference](#inference) | [DeepSpeed](#deepspeed) | [Adapter](#Adapter) | [Offsite-tuning](#offsite-tuning) |
#### LLM-general
|         Name          | (Type) Default Value | Description                                              | Note |
|:---------------------:|:--------------------:|:---------------------------------------------------------|:-----|
|   `cfg.llm.tok_len`   |      (int) 128       | Max token length for model input (training)              ||
| `cfg.llm.cache.model` |     (string) ''      | The fold for storing model cache, default in `~/.cache/` ||
|||||
#### Inference
|              Name              | (Type) Default Value | Description                                  | Note |
|:------------------------------:|:--------------------:|:---------------------------------------------|:-----|
|     `cfg.llm.chat.max_len`     |      (int) 1000      | Max token length for model input (inference) ||
| `cfg.llm.chat.max_history_len` |       (int) 10       | Max number of history texts                  ||
#### DeepSpeed
|             Name              | (Type) Default Value | Description                                                  | Note                                                                                                                                               |
|:-----------------------------:|:--------------------:|:-------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
|    `cfg.llm.deepspeed.use`    |     (bool) False     | Whether use DeepSpeed                                        | Use `nvcc - V` to make sure CUDA installed. When set it to `True`, we can full-parameter fine-tune a `llama-7b` on a machine with 4 V100-32G gpus. |
| `cfg.llm.deepspeed.ds_config` |     (string) ''      | The path to the file containing configurations for DeepSpeed | See `federatedscope/llm/baseline/deepspeed/ds_config.json`                                                                                         |
#### Adapter
|            Name             | (Type) Default Value | Description                                              | Note                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|:---------------------------:|:--------------------:|:---------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|    `cfg.llm.adapter.use`    |     (bool) False     | Whether use adapter                                      ||
|   `cfg.llm.adapter.args`    |     list ([{}])      | Args for adapters                                        | We offer the following four adaptets:<br/>`[ { 'adapter_package': 'peft', 'adapter_method': 'lora', 'r': 8, 'lora_alpha': 32, 'lora_dropout': 0.1 } ]`; <br/> `[{'adapter_package': 'peft', 'adapter_method': 'prefix', 'prefix_projection': False, 'num_virtual_tokens': 20}]`; <br/> `[{'adapter_package': 'peft', 'adapter_method': 'p-tuning', 'encoder_reparameterization_type': 'MLP', 'encoder_dropout': 0.1, 'num_virtual_tokens': 20}]`; <br/> `[{'adapter_package': 'peft', 'adapter_method': 'prompt', 'prompt_tuning_init': 'RANDOM', 'num_virtual_tokens': 20}]`. |
| `cfg.llm.adapter.mv_to_cpu` |     (bool) False     | Whether move the adapter to cpu after each training step | If true, it can save memory but cost more time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
#### Offsite-tuning
|                            Name                             |  (Type) Default Value  | Description                                                     | Note                                                                                                                                                                                   |
|:-----------------------------------------------------------:|:----------------------:|:----------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                `cfg.llm.offsite_tuning.use`                 |      (bool) False      | Whether apply offsite-tuning                                    | Set it `True` when clients cannot access to the full model                                                                                                                             |
|              `cfg.llm.offsite_tuning.strategy`              | (string) 'drop_layer'  | The mothod used for offsite-tuning                              | More methods will be supported ASAP                                                                                                                                                    |
|               `cfg.llm.offsite_tuning.emu_l`                |        (int) 1         | Fix the previous layers as adapter for training                 ||
|               `cfg.llm.offsite_tuning.emu_r`                |        (int) 10        | Fix the layers behind as adapter for training                   ||
|               `cfg.llm.offsite_tuning.kwargs`               |      (list) [{}]       | Args for offsite-tuning method                                  | E.g.,`[{'drop_ratio':0.2}]` means uniformly drops 20% of the layers between `cfg.llm.offsite_tuning.emu_l` and `cfg.llm.offsite_tuning.emu_r`, denote the remaining as emulator        |
|             `cfg.llm.offsite_tuning.eval_type`              |     (string) 'emu'     | The type of evaluation for offsite-tuning                       | 'full' means evaluating the original model with fine-tuned adapters; 'emu' means evaluating the emulator with fine-tuned adapters                                                      |
|           `cfg.llm.offsite_tuning.emu_align.use`            |      (bool) False      | Whether use model distillation                                  | If `True`, the server will regard the layers between `cfg.llm.offsite_tuning.emu_l` and `cfg.llm.offsite_tuning.emu_r` as a teacher model, and distill a student model as the emulator |
|       `cfg.llm.offsite_tuning.emu_align.restore_from`       |      (string) ''       | The path to the emulator load by clients to perform fine-tuning ||
|         `cfg.llm.offsite_tuning.emu_align.save_to`          |      (string) ''       | The path to the emulator saved by server                        ||
|     `cfg.llm.offsite_tuning.emu_align.exit_after_align`     |      (bool) False      | Whether exist after model distillation                          ||
|        `cfg.llm.offsite_tuning.emu_align.data.root`         |    (string) 'data'     | The fold where the `data` file located for model distilation    ||
|        `cfg.llm.offsite_tuning.emu_align.data.type`         | (string) 'alpaca@llm'  | The Dataset name for model distillation                         ||
|       `cfg.llm.offsite_tuning.emu_align.data.splits`        | (list) [0.8, 0.1, 0.1] | Train, valid, test splits for model distillation                ||
| `cfg.llm.offsite_tuning.emu_align.train.local_update_steps` |        (int) 10        | The number of local training steps in model distillation        ||
|   `cfg.llm.offsite_tuning.emu_align.train.batch_or_epoch`   |    (string) 'batch'    | The type of local training for model distillation               ||
|   `cfg.llm.offsite_tuning.emu_align.train.lm_loss_weight`   |      (float) 0.1       | The ratio of language model loss in model distillation          ||
|   `cfg.llm.offsite_tuning.emu_align.train.kd_loss_weight`   |      (float) 0.9       | The ratio of knowledge distillation loss in model distillation  ||
|   `cfg.llm.offsite_tuning.emu_align.train.optimizer.type`   |     (string) 'SGD'     | The type of optimizer used in model distillation                ||
|    `cfg.llm.offsite_tuning.emu_align.train.optimizer.lr`    |      (float) 0.01      | The learning rate used in model distillation                    ||

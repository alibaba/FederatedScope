## Configurations
We summarize all the customizable configurations here.

### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| `data.root` | (string) 'data' | <font size=1> The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. </font> | - |
| `data.type` | (string) 'toy' | <font size=1>Dataset name</font> | CV: 'femnist', 'celeba' ; NLP: 'shakespeare', 'subreddit', 'twitter'; Graph: 'cora', 'citeseer', 'pubmed', 'dblp_conf', 'dblp_org', 'csbm', 'epinions', 'ciao', 'fb15k-237', 'wn18', 'fb15k' , 'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1', 'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'IMDB-BINARY', 'IMDB-MULTI', 'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP', 'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX', 'graph_multi_domain_mol', 'graph_multi_domain_small', 'graph_multi_domain_mix', 'graph_multi_domain_biochem'; MF: 'vflmovielens1m', 'vflmovielens10m', 'hflmovielens1m', 'hflmovielens10m', 'vflnetflix', 'hflnetflix'; Tabular: 'toy', 'synthetic'; External dataset: 'DNAME@torchvision', 'DNAME@torchtext', 'DNAME@huggingface_datasets', 'DNAME@openml'. |
| `data.args` | (list) [] | <font size=1>Args for the external dataset</font> | Used for external dataset, eg. `[{'download': False}]` |
| `data.splitter` | (string) '' | <font size=1>Splitter name for standalone dataset</font> | Generic splitter: 'lda'; Graph splitter: 'louvain', 'random', 'rel_type', 'graph_type', 'scaffold', 'scaffold_lda', 'rand_chunk' |
| `data.splitter_args` | (list) [] | <font size=1>Args for splitter.</font> | Used for splitter, eg. `[{'alpha': 0.5}]` |
| `data.transform` | (list) [] | <font size=1>Transform for x of data</font> | Used in `get_item` in torch.dataset, eg. `[['ToTensor'], ['Normalize', {'mean': [0.1307], 'std': [0.3081]}]]` |
| `data.target_transform` | (list) [] | <font size=1>Transform for y of data</font> | Use as `data.transform` |
| `data.pre_transform` | (list) [] | <font size=1>Pre_transform for `torch_geometric` dataset</font> | Use as `data.transform` |
| `data.batch_size` | (int) 64 | <font size=1>batch_size for DataLoader</font> | - |
| `data.drop_last` | (bool) False | <font size=1>Whether drop last batch (if the number of last batch is smaller than batch_size) in DataLoader</font> | - |
| `data.sizes` | (list) [10, 5] | <font size=1>Sample size for graph DataLoader</font> | The length of `data.sizes` must meet the layer of GNN models. |
| `data.shuffle` | (bool) True | <font size=1>Shuffle train DataLoader</font> | - |
| `data.server_holds_all` | (bool) False | <font size=1>Only use in global mode, whether the server (workers with idx 0) holds all data, useful in global training/evaluation case</font> | - |
| `data.subsample` | (float) 1.0 | <font size=1> Only used in LEAF datasets, subsample clients from all clients</font> | - |
| `data.splits` | (list) [0.8, 0.1, 0.1] | <font size=1>Train, valid, test splits</font> | - |
| `data.consistent_label_distribution` | (bool) False | <font size=1>Make label distribution of train/val/test set over clients keep consistent during splitting</font> | - |
| `data.cSBM_phi` | (list) [0.5, 0.5, 0.5] | <font size=1>Phi for cSBM graph dataset</font> | - |
| `data.loader` | (string) '' | <font size=1>Graph sample name, used in minibatch trainer</font> | 'graphsaint-rw': use `GraphSAINTRandomWalkSampler` as DataLoader; 'neighbor': use `NeighborSampler` as DataLoader. |
| `data.num_workers` | (int) 0 | <font size=1>num_workers in DataLoader</font> | - |
| `data.graphsaint.walk_length` | (int) 2 | <font size=1>The length of each random walk in graphsaint.</font> | - |
| `data.graphsaint.num_steps` | (int) 30 | <font size=1>The number of iterations per epoch in graphsaint.</font> | - |
| `cfg.data.quadratic.dim` | (int) 1 | <font size=1>Dim of synthetic quadratic  dataset</font> | - |
| `cfg.data.quadratic.min_curv` | (float) 0.02 | <font size=1>Min_curve of synthetic quadratic  dataset</font> | - |
| `cfg.data.quadratic.max_curv` | (float) 12.5 | <font size=1>Max_cur of synthetic quadratic  dataset</font> | - |

### Federated training
The configurations related to federated training are defined in `cfg_training.py`.
Considering it's infeasible to list all the potential arguments for optimizers and schedulers, we allow the users to add new parameters directly under the corresponding namespace. 
For example, we haven't defined the argument `train.optimizer.weight_decay` in `cfg_training.py`, but the users are allowed directly use it. 
If the optimizer doesn't require the argument named `weight_decay`, an error will be raised. 

#### Local training
The following configurations are related to the local training. 

|            Name            | (Type) Default Value |                   Description                    |                                                                                         Note                                                                                         |
|:--------------------------:|:--------------------:|:------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `train.local_update_steps` |       (int) 1        |       The number of local training steps.        |                                                                                          -                                                                                           |
|   `train.batch_or_epoch`   |   (string) 'batch'   |           The type of local training.            |               `train.batch_or_epoch` specifies the unit that `train.local_update_steps` adopts. All new parameters will be used as arguments for the chosen optimizer.               |
|     `train.optimizer`      |          -           |                        -                         |                     You can add new parameters under `train.optimizer` according to the optimizer, e.g., you can set momentum by `cfg.train.optimizer.momentum`.                     |
|   `train.optimizer.type`   |    (string) 'SGD'    |  The type of optimizer used in local training.   |                                               Currently we support all optimizers build in PyTorch (The modules under `torch.optim`).                                                |
| `train.optimizer.lr` |     (float) 0.1      |  The learning rate used in the local training.   |                                                                                          -                                                                                           |
|     `train.scheduler`      |          -           |                        -                         | Similar with `train.optimizer`, you can add new parameters as you need, e.g., `train.scheduler.step_size=10`. All new parameters will be used as arguments for the chosen scheduler. |
| `train.scheduler.type` |     (string) ''      | The type of the scheduler used in local training |                                         Currently we support all schedulers build in PyTorch (The modules under `torch.optim.lr_scheduler`).                                         |

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
|   `early_step.the_smaller_the_better`    | (bool) True | The optimized direction of the chosen metric |                                                                                           -                                                                                           |

### FL Setting
The configurations related to FL settings are defined in `cfg_fl_setting.py`.
#### `federate`: basic fl setting
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
#### `distribute`: for distribute mode
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `distribute.use` | (bool) False | Whether to run FL courses with distribute mode. | If `False`, all the related configurations (`cfg.distribute.xxx`) would not take effect.  |
| `distribute.server_host` | (string) '0.0.0.0' | The host of server's ip address for communication | - |
| `distribute.server_port` | (string) 50050 | The port of server's ip address for communication | - |
| `distribute.client_host` | (string) '0.0.0.0' | The host of client's ip address for communication | - |
| `distribute.client_port` | (string) 50050 | The port of client's ip address for communication | - |
| `distribute.role` | (string) 'client' </br> Choices: {'server', 'client'} | The role of the worker | - |
| `distribute.data_file` | (string) 'data' | The path to the data dile | - |
| `distribute.data_idx` | (int) -1 | It is used to specify the data index in distributed mode when adopting a centralized dataset for simulation (formatted as {data_idx: data/dataloader}). | `data_idx=-1` means that the entire dataset is owned by the participant. And we randomly sample the index in simulation for other invalid values excepted for -1.
| `distribute.` </br>`grpc_max_send_message_length` | (int) 100 * 1024 * 1024 | The maximum length of sent messages | - |
| `distribute.` </br>`grpc_max_receive_message_length` | (int) 100 * 1024 * 1024 | The maximum length of received messages | - |
| `distribute.`grpc_enable_http_proxy | (bool) False | Whether to enable http proxy | - |
#### `vertical`: for vertical federated learning
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `vertical.use` | (bool) False | Whether to run vertical FL. | If `False`, all the related configurations (`cfg.vertical.xxx`) would not take effect.  |
| `vertical.encryption` | (string) `paillier` | The encryption algorithms used in vertical FL. | - |
| `vertical.dims` | (list of int) [5,10] | The dimensions of the input features for participants. | - |
| `vertical.key_size` | (int) 3072 | The length (bit) of the public keys. | - | 

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

|      Name       | (Type) Default Value | Description                        | Note                                                    |
|:---------------:|:--------------------:|:-----------------------------------|:--------------------------------------------------------|
|   `sgdmf.use`   |     (bool) False     | The indicator of the SGDMF method. |                                                         |
|    `sgdmf.R`    |      (float) 5.      | The upper bound of rating.         | -                                                       |
| `sgdmf.epsilon` |      (float) 4.      | The $\epsilon$ used in DP.         | -                                                       |
| `sgdmf.delta` |     (float) 0.5      | The $\delta$ used in DP.           | -                                                       |
| `sgdmf.constant` |      (float) 1. | The constant in SGDMF | -                                                       |
| `sgdmf.theta` | (int) -1 | - | -1 means per-rating privacy, otherwise per-user privacy |

### Auto-tuning Components

These arguments are exposed for customizing our provided auto-tuning components.

#### General

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
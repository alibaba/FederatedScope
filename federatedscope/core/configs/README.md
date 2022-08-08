## Configurations
We summarize all the customizable configurations here.

### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| data.root | (string) 'data' | <font size=1> The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. </font> | - |

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

### Asynchronous Training Strategies
The configurations related to applying asynchronous training strategies in FL are defined in `cfg_asyn.py`.

| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| asyn.use | (bool) False| Whether to use asynchronous training strategies. | If `False`, all the related configurations (`cfg.asyn.xxx`) would not task effect.  | - |
| asyn.time_budget | (int/float) 0 | The predefined time budget (seconds) for each training round. | `time_budget`<=0 means the time budget is not applied. |
| asyn.min_received_num | (int) 2 | The minimal number of received feedback for the server to trigger federated aggregation. | - |
| asyn.min_received_rate | (float) -1.0 | The minimal ratio of received feedback w.r.t. the sampled clients for the server to trigger federated aggregation. | - |
| asyn.staleness_toleration | (int) 0 | The threshold of the tolerable staleness in federated aggregation. | - | 
| asyn.staleness_discount_factor | (float) 1.0 | The discount factor for the staled feedback in federated aggregation. | - |
| asyn.aggregator | (string) 'goal_achieved' </br> Choices: {'goal_achieved', 'time_up'} | The condition for federated aggregation. | 'goal_achieved': perform aggregation when the defined number of feedback has been received; 'time_up': perform aggregation when the allocated time budget has been run out. |
| asyn.broadcast_manner | (string) 'after_aggregating' </br> Choices: {'after_aggregating', 'after_receiving'} | The broadcasting manner of server. | 'after_aggregating': broadcast the up-to-date global model after performing federated aggregation; 'after_receiving': broadcast the up-to-date global model after receiving the model update from clients. |
| asyn.overselection | (bool) False | Whether to use the overselection technique | - |

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

# SGDMF
The configurations related to SGDMF method (only used in matrix factorization tasks).

|      Name       | (Type) Default Value | Description                        | Note                                                    |
|:---------------:|:--------------------:|:-----------------------------------|:--------------------------------------------------------|
|   `sgdmf.use`   |     (bool) False     | The indicator of the SGDMF method. |                                                         |
|    `sgdmf.R`    |      (float) 5.      | The upper bound of rating.         | -                                                       |
| `sgdmf.epsilon` |      (float) 4.      | The $\epsilon$ used in DP.         | -                                                       |
| `sgdmf.delta` |     (float) 0.5      | The $\delta$ used in DP.           | -                                                       |
| `sgdmf.constant` |      (float) 1. | The constant in SGDMF | -                                                       |
| `sgdmf.theta` | (int) -1 | - | -1 means per-rating privacy, otherwise per-user privacy |

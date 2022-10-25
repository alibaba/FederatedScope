# Local Learning Abstraction: Trainer

FederatedScope decouples the local learning process and details of FL communication and schedule, allowing users to freely customize the local learning algorithm via the `trainer`. Each worker holds a `trainer` object to manage the details of local learning, such as the loss function, optimizer, training step, evaluation, etc.

This tutorial is a shorter version of [full version tutorial](https://federatedscope.io/docs/trainer/), where you can learn more details about FS Trainer.

## Code Structure

The code structure is shown below, and we will discuss all the concepts of our FS Trainer later.

```bash
federatedscope/core
├── trainers
│   ├── BaseTrainer
│   │   ├── Trainer
│   │   │   ├── GeneralTorchTrainer
│   │   │   ├── GeneralTFTrainer
│   │   │   ├── Context
│   │   │   ├── ...
│   │   ├── UserDefineTrainer
│   │   ├── ...
```

## FS Trainer

A typical machine-learning process consists of the following procedures:

1. Preparing data.
2. Iterations over training datasets to update the model parameters
3. Evaluation of the quality of the learned model on validation/evaluation datasets
4. Saving, loading, and monitoring the model and intermediate results

### BaseTrainer

`BaseTrainer` is an abstract class of our Trainer, which provide the interface of each method. And you can implement your own trainer by inheriting from `BaseTrainer`. More examples can be found in `federatedscope/contrib/trainer`.

```python
class BaseTrainer(abc.ABC):
    def __init__(self, model, data, device, **kwargs):
        self.model = model
        self.data = data
        self.device = device
        self.kwargs = kwargs

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, target_data_split_name='test'):
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, model_parameters, strict=False):
        raise NotImplementedError

    @abc.abstractmethod
    def get_model_para(self):
        raise NotImplementedError
    
    ... ...
```

### Trainer

As the figure shows, in FederatedScope `Trainer` (a subclass of `BaseTrainer`), these above procedures are provided with high-level `routines` abstraction, which is made up of `Context` class and several pluggable `Hooks`. And we provide `GeneralTorchTrainer` and `GeneralTFTrainer` for `PyTorch` and `TensorFlow`, separately.

<img src="https://img.alicdn.com/imgextra/i4/O1CN01H8OEeS1tdhR38C4dK_!!6000000005925-2-tps-1504-874.png" alt="undefined" style="zoom:50%;" />

#### Context

The `Context` class (a subclass of `dict`) is used to hold learning-related attributes, including data, model, optimizer and etc, and user and add or delete these attributes in hook functions. We classify and show the default attributes below:

* Data-related attributes
  * `ctx.data`: the raw data (not split) the trainer holds
  * `ctx.num_samples`: the number of samples used in training
  * `ctx.train_data`, `ctx.val_data`, `ctx.test_data`: the split data the trainer holds
  * `ctx.train_loader`, `ctx.val_loader`, `ctx.test_loader`: the DataLoader of each split data
  * `ctx.num_train_data`, `ctx.num_val_data`, `ctx.num_test_data`: the number of samples of  the split data
* Model-related attributes
  * `ctx.model`: the model used
  * `ctx.models`: the multi models if use
  * `ctx.mirrored_models`: the mirrored models
  * `ctx.trainable_para_names`: the trainable parameter names of the model
* Optimizer-related attributes
  * `ctx.optimizer`: see [`torch.optim`](https://pytorch.org/docs/stable/optim.html#module-torch.optim) for details
  * `ctx.scheduler`: decays the learning rate of each parameter group
  * `ctx.criterion`: loss/criterion function
  * `ctx.regularizer`: regular terms
  * `ctx.grad_clip`: gradient clipping
* Mode-related attributes
  * `ctx.cur_mode`: mode of trainer, which is one of `['train', 'val', 'test']`
  * `ctx.mode_stack`: stack of mode, only used for switching mode 
  * `ctx.cur_split`: split of data, which is one of `['train', 'val', 'test']` (Note: use `train` data in `test` mode is allowed)
  * `ctx.split_stack`: stack of split, only used for switching data split 
* Metric-related attributes
  * `ctx.loss_batch_total`: Loss of current batch
  * `ctx.loss_regular_total`: Loss of regular term 
  * `ctx.y_true`:  true label of batch data
  * `ctx.y_prob`: output of the model with batch data as input
  * `ctx.ys_true`: true label of data
  * `ctx.ys_prob`: output of the model
  * `ctx.eval_metrics`: evaluation metrics calculated by `Monitor`
  * `ctx.monitor`: used for monitor trainer's behavior and statistics
* Other (statistics) attributes (@property, query from ``cfg`` if not set)
  * `ctx.cfg`: configuration of FL course, see [link](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/core/configs) for details
  * `ctx.device`: current device, such as `cpu` and `gpu0`.
  * `ctx.num_train_batch_last_epoch`, `ctx.num_total_train_batch`: the number of batch
  * `ctx.num_train_epoch`, `ctx.num_val_epoch`, `ctx.num_test_epoch`: the number of epoch in each data split
  * `ctx.num_train_batch`, `ctx.num_val_batch`, `ctx.num_test_batch`: the number of batch in each data split

#### Hooks

The `Hooks` represent fine-grained learning behaviors at different point-in-times, which provides a simple yet powerful way to customize learning behaviors with a few modifications and easy re-use of fruitful default hooks. In this section, we will show the detail of each hook used in `Trainer`.

##### Hook trigger

The hook trigger is where the hook functions are executed,  and all the hook functions are executed following the pattern below:

* **on_fit_start**
  * **on_epoch_start**
    * **on_batch_start**
    * **on_batch_forward**
    * **on_batch_backward**
    * **on_batch_end**
  * **on_epoch_end**
* **on_fit_end**

##### Train hooks

Train hooks are executed when `ctx.cur_mode` is `train`, following the execution paradigm as shown below:

* **on_fit_start**

  `_hook_on_fit_start_init`

  `_hook_on_fit_start_calculate_model_size`

  * **on_epoch_start**

    `_hook_on_epoch_start`

    * **on_batch_start**

      `_hook_on_batch_start_init`

    * **on_batch_forward**

      `_hook_on_batch_forward`

      `_hook_on_batch_forward_regularizer`

      `_hook_on_batch_forward_flop_count`

    * **on_batch_backward**

      `_hook_on_batch_backward`

    * **on_batch_end**

      `_hook_on_batch_end`

  * **on_epoch_end**

    `None`

* **on_fit_end**

  `_hook_on_fit_end`

##### Evaluation (val/test) hooks

Evaluation hooks are executed when `ctx.cur_mode` is `val` or `test`, following the execution paradigm as shown below:

* **on_fit_start**

  `_hook_on_fit_start_init`

  * **on_epoch_start**

    `_hook_on_epoch_start`

    * **on_batch_start**

      `_hook_on_batch_start_init`

    * **on_batch_forward**

      `_hook_on_batch_forward`

    * **on_batch_backward**

      `None`

    * **on_batch_end**

      `_hook_on_batch_end`

  * **on_epoch_end**

    `None`

* **on_fit_end**

  `_hook_on_fit_end`

##### Finetune hooks

Finetune hooks are executed when `ctx.cur_mode` is `finetune`, following the execution paradigm as shown below:

* **on_fit_start**

  `_hook_on_fit_start_init`

  `_hook_on_fit_start_calculate_model_size`

  * **on_epoch_start**

    `_hook_on_epoch_start`

    * **on_batch_start**

      `_hook_on_batch_start_init`

    * **on_batch_forward**

      `_hook_on_batch_forward`

      `_hook_on_batch_forward_regularizer`

      `_hook_on_batch_forward_flop_count`

    * **on_batch_backward**

      `_hook_on_batch_backward`

    * **on_batch_end**

      `_hook_on_batch_end`

  * **on_epoch_end**

    `None`

* **on_fit_end**

  `_hook_on_fit_end`

##### Hook functions

In this section, we will briefly describe what the hook functions do with the attributes/variables in `ctx`.

###### GeneralTorchTrainer

* `_hook_on_fit_start_init`

  | Modified attribute       | Operation               |
  | ------------------------ | ----------------------- |
  | `ctx.model`              | Move to `ctx.device`    |
  | `ctx.optimizer`          | Initialize by `ctx.cfg` |
  | `ctx.scheduler`          | Initialize by `ctx.cfg` |
  | `ctx.loss_batch_total`   | Initialize to `0`       |
  | `ctx.loss_regular_total` | Initialize to `0`       |
  | `ctx.num_samples`        | Initialize to `0`       |
  | `ctx.ys_true`            | Initialize to `[]`      |
  | `ctx.ys_prob`            | Initialize to `[]`      |

* `_hook_on_fit_start_calculate_model_size`

  | Modified attribute | Operation        |
  | ------------------ | ---------------- |
  | `ctx.monitor`      | Track model size |

* `_hook_on_epoch_start`

  | Modified attribute           | Operation             |
  | ---------------------------- | --------------------- |
  | `ctx.{ctx.cur_split}_loader` | Initialize DataLoader |

* `_hook_on_batch_start_init` 

  | Modified attribute | Operation             |
  | ------------------ | --------------------- |
  | `ctx.data_batch`   | Initialize batch data |

* `_hook_on_batch_forward`

  | Modified attribute | Operation                           |
  | ------------------ | ----------------------------------- |
  | `ctx.y_true`       | Move to `ctx.device`                |
  | `ctx.y_prob`       | Forward propagation to get `y_prob` |
  | `ctx.loss_batch`   | Calculate the loss                  |
  | `ctx.batch_size`   | Get the batch_size                  |

* `_hook_on_batch_forward_regularizer`

  | Modified attribute | Operation                                 |
  | ------------------ | ----------------------------------------- |
  | `ctx.loss_regular` | Calculate the regular loss                |
  | `ctx.loss_task`    | Sum the `ctx.loss_regular` and `ctx.loss` |

* `_hook_on_batch_forward_flop_count`

  | Modified attribute | Operation           |
  | ------------------ | ------------------- |
  | `ctx.monitor`      | Track average flops |

* `_hook_on_batch_backward`

  | Modified attribute | Operation            |
  | ------------------ | -------------------- |
  | `ctx.optimizer`    | Update by gradient   |
  | `ctx.loss_task`    | Backward propagation |
  | `ctx.scheduler`    | Update by gradient   |

* `_hook_on_batch_end `

  | Modified attribute       | Operation              |
  | ------------------------ | ---------------------- |
  | `ctx.num_samples`        | Add `ctx.batch_size`   |
  | `ctx.loss_batch_total`   | Add batch loss         |
  | `ctx.loss_regular_total` | Add batch regular loss |
  | `ctx.ys_true`            | Append `ctx.y_true`    |
  | `ctx.ys_prob`            | Append `ctx.ys_prob`   |

* `_hook_on_fit_end `

  | Modified attribute | Operation                                |
  | ------------------ | ---------------------------------------- |
  | `ctx.ys_true`      | Convert to `numpy.array`                 |
  | `ctx.ys_prob`      | Convert to `numpy.array`                 |
  | `ctx.monitor`      | Evaluate the results                     |
  | `ctx.eval_metrics` | Get evaluated results from `ctx.monitor` |

###### DittoTrainer

* `_hook_on_fit_start_set_regularized_para`

  | Modified attribute               | Operation                                                    |
  | -------------------------------- | ------------------------------------------------------------ |
  | `ctx.global_model`               | Move to `ctx.device` and set to `train` mode                 |
  | `ctx.local_model`                | Move to `ctx.device` and set to `train` mode                 |
  | `ctx.optimizer_for_global_model` | Initialize by `ctx.cfg` and wrapped by `wrap_regularized_optimizer` |
  | `ctx.optimizer_for_local_model`  | Initialize by `ctx.cfg` and set compared parameter group     |

* `_hook_on_fit_start_clean`

  | Modified attribute                  | Operation         |
  | ----------------------------------- | ----------------- |
  | `ctx.optimizer`                     | Delete            |
  | `ctx.num_samples_local_model_train` | Initialize to `0` |

* `_hook_on_fit_start_switch_local_model`

  | Modified attribute | Operation                                       |
  | ------------------ | ----------------------------------------------- |
  | `ctx.model`        | Set to `ctx.local_model` and set to `eval` mode |

* `_hook_on_batch_start_switch_model`

  | Modified attribute            | Operation                                                    |
  | ----------------------------- | ------------------------------------------------------------ |
  | `ctx.use_local_model_current` | Set to `True` or `False`                                     |
  | `ctx.model`                   | Set to `ctx.local_model` or `ctx.global_model`               |
  | `ctx.optimizer`               | Set to `ctx.optimizer_for_local_model` or `ctx.optimizer_for_global_model` |

* `_hook_on_batch_forward_cnt_num`

  | Modified attribute                  | Operation            |
  | ----------------------------------- | -------------------- |
  | `ctx.num_samples_local_model_train` | Add `ctx.batch_size` |

* `_hook_on_batch_end_flop_count`

  | Modified attribute | Operation           |
  | ------------------ | ------------------- |
  | `ctx.monitor`      | Monitor total flops |

* `_hook_on_fit_end_calibrate`

  | Modified attribute | Operation                                          |
  | ------------------ | -------------------------------------------------- |
  | `ctx.num_samples`  | Minus `ctx.num_samples_local_model_train`          |
  | `ctx.eval_metrics` | Record `train_total` and `train_total_local_model` |

* `_hook_on_fit_end_switch_global_model`

  | Modified attribute | Operation                 |
  | ------------------ | ------------------------- |
  | `ctx.model `       | Set to `ctx.global_model` |

* `_hook_on_fit_end_free_cuda`

  | Modified attribute | Operation     |
  | ------------------ | ------------- |
  | `ctx.global_model` | Move to `cpu` |
  | `ctx.local_model`  | Move to `cpu` |

###### pFedMeTrainer

* `_hook_on_fit_start_set_local_para_tmp`

  | Modified attribute           | Operation                                                    |
  | ---------------------------- | ------------------------------------------------------------ |
  | `ctx.optimizer`              | Wrapped by `wrap_regularized_optimizer` and set compared parameter group |
  | `ctx.pFedMe_outer_lr`        | Initialize to `ctx.cfg.train.optimizer.lr`                   |
  | `ctx.pFedMe_local_model_tmp` | Copy from `ctx.model`                                        |

* `_hook_on_batch_start_init_pfedme`

  | Modified attribute              | Operation                          |
  | ------------------------------- | ---------------------------------- |
  | `ctx.data_batch_cache`          | Copy from `ctx.data_batch`         |
  | `ctx.pFedMe_approx_fit_counter` | Count to refresh data every K step |

* `_hook_on_batch_end_flop_count`

  | Modified attribute | Operation           |
  | ------------------ | ------------------- |
  | `ctx.monitor`      | Monitor total flops |

* `_hook_on_epoch_end_flop_count`

  | Modified attribute | Operation           |
  | ------------------ | ------------------- |
  | `ctx.monitor`      | Monitor total flops |

* `_hook_on_epoch_end_update_local`

  | Modified attribute | Operation                                         |
  | ------------------ | ------------------------------------------------- |
  | `ctx.model`        | Update parameters by `ctx.pFedMe_local_model_tmp` |
  | `ctx.optimizer`    | Set compared parameter group                      |

* `_hook_on_fit_end_update_local`

  | Modified attribute           | Operation                                         |
  | ---------------------------- | ------------------------------------------------- |
  | `ctx.model`                  | Update parameters by `ctx.pFedMe_local_model_tmp` |
  | `ctx.pFedMe_local_model_tmp` | Delete                                            |

###### FedProxTrainer & NbaflTrainer

* `_hook_record_initialization`

  | Modified attribute | Operation             |
  | ------------------ | --------------------- |
  | `ctx.weight_init`  | Copy from `ctx.model` |

* `_hook_del_initialization`

  | Modified attribute | Operation     |
  | ------------------ | ------------- |
  | `ctx.weight_init`  | Set to `None` |

* `_hook_inject_noise_in_upload`

  | Modified attribute | Operation                  |
  | ------------------ | -------------------------- |
  | `ctx.model`        | Inject noise to parameters |


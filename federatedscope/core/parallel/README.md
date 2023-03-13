# Parallelization for standalone mode

To facilitate developers to quickly verify their algorithms, we designed and implemented `StandaloneMultiGPURunner` with torch distributed data parallel (DDP). The new runner can better utilize the computing resources of multiple GPUs and accelerate training in standalone mode of FederatedScope.

## When to use
Use `StandaloneMultiGPURunner` when you have **multiple GPUs (>=2)** in your machine and need quick verification with **standalone mode**.


## Configuration

Add `federate.process_num` item in the configuration file to parallelize the training.

> Note: `federate.process_num` only takes effect when `use_gpu=True`, `backend='torch'`, `federate.mode='standalone'` and `federate.share_local_model=False`, and the value is required to be not greater than the number of GPUs.

```yaml
use_gpu: True
backend: 'torch'
device: 0
early_stop:
  patience: 5
seed: 12345
federate:
  mode: standalone
  client_num: 100
  total_round_num: 20
  sample_client_rate: 0.2
  share_local_model: False
  process_num: 4 # run 4 processes simultaneously
...
```

## Use cases

Here we give an example to demonstrate the efficiency of `StandaloneMultiGPURunner` compared to `StandaloneRunner`. The configuration file and experiment result are listed below.
The experiment result shows that the totoal running time of `StandaloneMultiGPURunner` is only 1/3 of `StandaloneRunner` in the case of 8 GPUs.

```yaml
use_gpu: True
device: 0
early_stop:
  patience: 5
seed: 12345
federate:
  mode: standalone
  client_num: 100
  total_round_num: 10
  sample_client_rate: 0.4
  share_local_model: False
  # use StandaloneMultiGPURunner with 8 GPUs
  process_num: 8
  # use StandaloneRunner
  # process_num: 1

data:
  root: data/
  type: femnist
  splits: [0.6,0.2,0.2]
  batch_size: 10
  subsample: 0.05
  num_workers: 0
  transform: [['ToTensor'], ['Normalize', {'mean': [0.1307], 'std': [0.3081]}]]
model:
  type: convnet2
  hidden: 2048
  out_channels: 62
train:
  local_update_steps: 1
  batch_or_epoch: epoch
  optimizer:
    lr: 0.01
    weight_decay: 0.0
grad:
  grad_clip: 5.0
criterion:
  type: CrossEntropyLoss
trainer:
  type: cvtrainer
eval:
  freq: 10
  metrics: ['acc', 'correct']
```

|  | StandaloneMultiGPURunner | StandaloneRunner |
| :---: | :---: | :---: |
| Total running time (minute) | 0.2406 | 0.7292 |
# FedHPO-B

A benchmark suite for studying federated hyperparameter optimization. FedHPO-B incorporates comprehensive FL tasks, enables efficient function evaluations, and eases continuing extensions. We also conduct extensive experiments based on FedHPO-B to benchmark a few HPO methods.

## Quick Start

We highly recommend running FedHPO-B with conda.

### Step 0. Dependency

* FedHPO-B is built on [FederatedScope](https://github.com/alibaba/FederatedScope), please see [Installation](https://github.com/alibaba/FederatedScope#step-1-installation) for install FederatedScope.

* <u>(Optianal)</u> In order to reproduce the results in our paper, please consider installing the following packages via:

  ```bash
  # hpbandster, smac3, optuna via conda
  conda install hpbandster smac3 optuna -c conda-forge
  
  # dehb via git
  conda install dask distributed -c conda-forge
  git clone https://github.com/automl/DEHB.git
  cd DEHB
  git checkout b8dcba7b38bf6e7fc8ce3e84ea567b66132e0eb5
  export PYTHONPATH=~/DEHB:$PYTHONPATH
  ```

### Step 1. Installation

We recommend installing FedHPOB directly using git by:

```bash
git clone https://github.com/alibaba/FederatedScope.git
cd FederatedScope/benchmark/FedHPOB
export PYTHONPATH=~/FedHPOB:$PYTHONPATH
```

### Step 2. Prepare data files

**Note**: If you only want to use FedHPO-B, you can skip to **Step3**.



### Step3. Start running

```python
from fedhpob.benchmarks import TabularBenchmark

benchmark = TabularBenchmark('cnn', 'femnist', 'avg')

# get hyperparameters space
config_space = benchmark.get_configuration_space()

# get fidelity space
fidelity_space = benchmark.get_fidelity_space()

# get results
res = benchmark(config_space.sapmle_configuration(),
                fidelity_space.sapmle_configuration(),
                seed=1)
```

## Reproduce the results in our paper

We take Figure 11 as an example.

* First get best seen value of each optimizer, the results are stored in the `~/exp_results` by default.

  ```
  cd scripts/exp
  bash run_mode.sh cora raw gcn 0 avg
  bash run_mode.sh citeseer raw gcn 1 avg
  bash run_mode.sh pubmed raw gcn 1 avg
  ```

* Then draw the figure with tools we provide, the figures will be saved in `~/figures`.

  ```python
  from fedhpob.utils.draw import rank_over_time
  
  rank_over_time('exp_results', 'gcn', algo='avg', loss=False)
  ```

## Publications

If you find FedHPO-B useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2206.03966):

```tex
@article{Wang2022FedHPOBAB,
  title={FedHPO-B: A Benchmark Suite for Federated Hyperparameter Optimization},
  author={Zhen Wang and Weirui Kuang and Ce Zhang and Bolin Ding and Yaliang Li},
  journal={ArXiv},
  year={2022},
  volume={abs/2206.03966}
}
```
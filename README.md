# FedKSeed
**The codes will be released soon after internal review and collation.**

This repository contains the official implementation for the work “**Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes**”. See more details in our [paper](https://arxiv.org/abs/2312.06353).

> Pre-trained large language models (LLMs) require fine-tuning to improve their responsiveness to natural language instructions. Federated learning (FL) offers a way to perform fine-tuning using the abundant data on end devices without compromising data privacy. Most existing federated fine-tuning methods for LLMs rely on parameter-efficient fine-tuning techniques, which may not reach the performance heights possible with full-parameter tuning. However, the communication overhead associated with full-parameter tuning is prohibitively high for both servers and clients. This work introduces FedKSeed, a novel approach that employs zeroth-order optimization (ZOO) with a set of random seeds. It enables federated full-parameter tuning of billion-sized LLMs directly on devices. Our method significantly reduces transmission requirements between the server and clients to just a few scalar gradients and random seeds, amounting to only a few thousand bytes. Building on this, we develop a strategy to assess the significance of ZOO perturbations for FL, allowing for probability-differentiated seed sampling. This prioritizes perturbations that have a greater impact on model accuracy. Experiments across six scenarios with different LLMs, datasets and data partitions demonstrate that our approach outperforms existing federated LLM fine-tuning methods in terms of both communication efficiency and new task generalization.

The purpose of this implementation is to  evaluate the accuracy of our approach. To reduce the time consumed by the experiments, we did not have all clients compute the global model in each round. Instead, we performed such a computation only once, and allowed all clients to copy this model directly using `copy.deepcopy()` (please refer to line 145 of `main.py`).

## Project Structure
```Markdown
.
├── optimizers
│   ├── mezo_bias_optimizer.py  // implementation of FedKSeed-Pro
│   └── mezo_optimizer.py  // implementation of FedKSeed
├── utils_data
│   ├── default_tokens.py  // definitions of some special tokens
│   ├── llm_dataset.py  // utilities to load Dolly-15K
│   ├── load_data.py  // entrance to get dataloaders
│   ├── natural_instruction_loader.py  // utilities to load Natural Instructions
│   └── partition_data.py  // utilities to partition datasets in Dirichlet distribution
├── client.py
├── evaluations.py
├── main.py
└── server.py
```

## Requirements
Please see `requirements.txt`.

## Data Preparation
1. Natural Instructions
To run experiments on [Natural Instructions](https://github.com/allenai/natural-instructions), you need to unzip the downloaded dataset in directory `./data`.

2. Dolly-15K
To run experiments on [Dolly-15K](https://github.com/databrickslabs/dolly), you need to download the corresponding dataset in directory `./data`, with its name as `databricks-dolly-15k.jsonl`.

## Running Examples
We provide some example scripts to conduct the experiments. 
The arguments can be adjusted according to the `help` information in their definitions.
1. FedKSeed on Natural Instructions
```Shell
# On Natural Instructions, the number of clients `num_clients` does not require manual setting. 
# It will be automatically adjusted to the number of tasks in `splits/default/train_tasks.txt`.
python main.py --rounds 40 --model datajuicer/LLaMA-1B-dj-refine-150B --use_prompts --dataset instruct --lr 0.0000003 -K 1024 -m 0.05 --log
```

2. FedKSeed on Dolly-15K with $\alpha=0.5$
```Shell
python main.py --rounds 60 --model datajuicer/LLaMA-1B-dj-refine-150B --use_prompts --dataset dolly --iid dir0.5 --num_clients 200 --lr 0.0000003 -K 1024 -m 0.05 --log
```


3. FedKSeed-Pro on Natural Instructions
```Shell
# On Natural Instructions, the number of clients `num_clients` does not require manual setting. 
# It will be automatically adjusted to the number of tasks in `splits/default/train_tasks.txt`.
python main.py --rounds 40 --bias_sampling  --model datajuicer/LLaMA-1B-dj-refine-150B --use_prompts --dataset instruct --lr 0.0000003 -K 1024 -m 0.05 --log
```

4. FedKSeed-Pro on Dolly-15K with $\alpha=0.5$
```Shell
python main.py --rounds 60 --bias_sampling  --model datajuicer/LLaMA-1B-dj-refine-150B --use_prompts --dataset dolly --iid dir0.5 --num_clients 200 --lr 0.0000003 -K 1024 -m 0.05 --log
```

## License
This project adopts the Apache-2.0 License. 
If the implementations and/or our paper were useful to you, please consider citing this [work](https://arxiv.org/abs/2312.06353):
```latex
@article{qin2023federated,
      title={Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes}, 
      author={Zhen Qin and Daoyuan Chen and Bingchen Qian and Bolin Ding and Yaliang Li and Shuiguang Deng},
      journal={arXiv preprint arXiv:2312.06353}
      year={2023}
}
```


# FlexLoRA

This branch contains the official implementation for the work “**Federated Fine-tuning of Large Language Models under Heterogeneous Language Tasks and Client Resources**”. See more details in our [paper](https://arxiv.org/pdf/2402.11505.pdf). 

> Federated Learning (FL) has recently been applied to the parameter-efficient fine-tuning of Large Language Models (LLMs). While promising, it raises significant challenges due to the heterogeneous resources and data distributions of clients. This study introduces FlexLoRA, a simple yet effective aggregation scheme for LLM fine-tuning, which mitigates the “buckets effect” in traditional FL that restricts the potential of clients with ample resources by tying them to the capabilities of the least-resourced participants. FlexLoRA allows for dynamic adjustment of local LoRA ranks, fostering the development of a global model imbued with broader, less task-specific knowledge. By synthesizing a full-size LoRA weight from individual client contributions and employing Singular Value Decomposition (SVD) for weight redistribution, FlexLoRA fully leverages heterogeneous client resources. Involving over 1,600 clients performing diverse NLP tasks, our experiments validate the efficacy of FlexLoRA, with the federated global model achieving up to a 3.1% average improvement in downstream NLP task performance. FlexLoRA’s practicality is further underscored by its seamless integration with existing LoRA-based FL methods and theoretical analysis, offering a path toward scalable, privacy-preserving federated tuning for LLMs.

In the future, we will merge this branch into the [llm](https://github.com/alibaba/FederatedScope/tree/llm) branch of FederatedScope.

## Project Structure
```Markdown
.
├── fed_utils
│   ├── adaptive_peft.py  
│   ├── client.py  // local client for training data
│   ├── client_participation_scheduling.py  // select clients to particiate for each round
│   └── model_aggregation.py  // define aggregation methods 
├── templates // templates for generating prompt
├── utils
│   ├── callbacks.py  
│   └── prompter.py  
├── heterolora.py // experiments related to heteroLoRA
├── main.py // experiments related to FlexLoRA
```

## Requirements and Dependencies

Please install necessary packages throught the following command:

`pip install -r requirements.txt`

Also, please install the huggingface evaluate package through git source code：

`git clone https://github.com/huggingface/evaluate.git`

`cd evaluate`

`pip install -e .`

## Data Preparation

The data is collected from [Natural Instructions](https://github.com/allenai/natural-instructions) and subsampled 10% randomly within each individual task. Inside the directory `./data`, the data is subsampled and partitioned into train/validdation/test sets for each client. Please refer to our paper for more details about the data preparation techniques that we used for experimentation.

## Arguments for Experiment：

`global_model`: pretrained model path(e.g. LLAMA)

`data_path` : data path

`cache_dir` : directory to cache your dataset

`output_dir`: directory to save the trained model in each commm round. 

`session_name`: name your experiment session

`seed`: random seed. Default: 42



#### FL hyperparamas

`client_selection_frac`: fraction of participation clients.

`num_communication_rounds`: number of total communication rounds

`num_clients`: number of total clients. For natural instruction META split, the client number is 1613

`resume_epoch`: the commucation round to resume training. If not continue training, can be set into `None`

`aggregation`: aggregation method. Current supports: `'homo'` -> homogenerous rank distribution, baseline; `'random'` -> uniformly select different LoRA configuration for each client, `'heavy_tail_strong'` -> 85% of the client has largest LoRA configuration, and the rest clients uniformly select the rest LoRA configurations, `'heavy_tail'` -> 85% of the client has smallest LoRA configuration, and the rest clients uniformly select the rest LoRA configurations, `'normal'` -> clients rank distribution follows normal distribution

`baseline`: FL baseline method to incorporate FlexLoRA. Current supports: `'fedavg', 'fedit', 'slora'`

`R_1` : Parameter for SLoRA. Total number of rounds for stage 1 sparse finetuning.

`early_stop` : Early stop for FL training. If True, will apply early stop.

`patience` : Early stop patience.


#### Local training hyperparams

`local_batch_size`: Local client batch size

`local_micro_batch_size`: Local client micro batch size. The gradient accumulation is calculated by `local_batch_size / local_micro_batch_size`

`local_num_epochs`: Local epoch for each client

`local_learning_rate`: Local training learning rate

`cutoff_len`: cutoff length for token

`warmup`: warmup steps for local training

`lr_decay`: Learning rate decay. If true, will divide learning rate by 2 after 15-th comm round

`train_on_inputs`: whether to train model on input text

`group_by_length`: whether to group by length when training

`prompt_template_name`: The prompt template to use, will default to `'alpaca'`

#### LoRA hyperparams

`lora_r`: rank for LoRA initialization. For FedAvg, need to set it to 8. For other settings, this hyperparameter can be set into any random value. 

`lora_alpha`: lora_alpha for LoRA 

`lora_dropout`: lora_dropout for LoRA

`lora_target_modules`: layers to put LoRA on




## Running Examples
We provide some example scripts to conduct the experiments. 
The arguments can be adjusted according to the `help` information in their definitions.
1. FedAvg without FlexLoRA(homogeneous rank distribution)
```Shell
python main.py --model datajuicer/LLaMA-1B-dj-refine-150B --baseline fedavg --aggregation homo --data_path ./data
```

2. FedAvg with FlexLoRA(normal rank distribution)
```Shell
python main.py --model datajuicer/LLaMA-1B-dj-refine-150B --baseline fedavg --aggregation normal --data_path ./data
```


3. SLoRA with FlexLoRA(normal rank distribution)
```Shell
python main.py --model datajuicer/LLaMA-1B-dj-refine-150B --baseline slora --aggregation normal --data_path ./data
```

4. FedAvg with HeteroLoRA(normal rank distribution)
```Shell
python heterolora.py --model datajuicer/LLaMA-1B-dj-refine-150B --baseline fedavg --aggregation normal --data_path ./data
```

## License
This project adopts the Apache-2.0 License. 
If the implementations and/or our paper were useful to you, please consider citing this [work](https://arxiv.org/pdf/2402.11505.pdf):
```latex
@article{bai2024federated,
  title={Federated Fine-tuning of Large Language Models under Heterogeneous Language Tasks and Client Resources},
  author={Bai, Jiamu and Chen, Daoyuan and Qian, Bingchen and Yao, Liuyi and Li, Yaliang},
  journal={arXiv preprint arXiv:2402.11505},
  year={2024}
}
```
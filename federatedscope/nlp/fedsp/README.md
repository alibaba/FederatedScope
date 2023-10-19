## Tunable Soft Prompts are Messengers in Federated Learning
The implementation of *Tunable Soft Prompts are Messengers in Federated Learning*.

In this study, we propose a novel FL training approach, named FedSP, that accomplishes information exchange among participants via tunable soft prompts. 
These soft prompts, updated and transmitted between the server and clients, assume the role of the global model parameters and serve as messengers to deliver useful knowledge from the local data and global model.

### Installation
First of all, you need to install FederatedScope, please refer to [installation](https://github.com/alibaba/FederatedScope#step-1-installation).

Besides, we need some additional requirements for NLP tasks, including:
* Transformers
* Datasets
* lm-eval

```bash
pip install transformers==4.21.0
pip install datasets
pip install lm-eval
```

### Reproduction
**Prefix-tuning**
```bash
bash run_gpt_prefix.sh $DEVICE  # gpt2-xl
bash run_opt_prefix.sh $DEVICE  # opt-1.3b
```

**FedSP**
```bash
bash run_gpt_fedsp.sh $DEVICE  # gpt2-xl
bash run_opt_fedsp.sh $DEVICE  # opt-1.3b
```

## Tunable Soft Prompts are Messengers in Federated Learning
The implementation of *Tunable Soft Prompts are Messengers in Federated Learning*.

In this study, we propose a novel FL training approach that accomplishes information exchange among participants via tunable soft prompts.
These soft prompts are updated and transmitted between the server and clients, taking over the duty of the global model parameters and serving as messengers to deliver useful knowledge in the local data and global model.

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

**FedPrompt**
```bash
bash run_gpt_fedprompt.sh $DEVICE  # gpt2-xl
bash run_opt_fedprompt.sh $DEVICE  # opt-1.3b
```

**FedPrompt-LSR**
```bash
bash run_gpt_fedprompt_lsr.sh $DEVICE  # gpt2-xl
bash run_opt_fedprompt_lsr.sh $DEVICE  # opt-1.3b
```

**FedSP (ours)**
```bash
bash run_gpt_fedsp.sh $DEVICE  # gpt2-xl
bash run_opt_fedsp.sh $DEVICE  # opt-1.3b
```

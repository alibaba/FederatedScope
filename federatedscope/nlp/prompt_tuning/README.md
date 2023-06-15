## Tunable Soft Prompts are Messengers in Federated Learning
The implementation of *Tunable Soft Prompts are Messengers in Federated Learning*.


### Installation
First of all, you need to install FederatedScope, please refer to [installation](https://github.com/alibaba/FederatedScope#step-1-installation).

Besides, we need some additional requirements for NLP tasks, including:
* transformers
* datasets
* lm-eval

```bash
pip install transformers==4.21.0
pip install datasets
pip install lm-eval
```

### Reproduction
**Prefix-Tuning**
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

**Ours**
```bash
bash run_gpt_ours.sh $DEVICE  # gpt2-xl
bash run_opt_ours.sh $DEVICE  # opt-1.3b
```

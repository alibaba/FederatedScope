set -e

wandb sweep sweep_fedAvg_FT.yaml
wandb sweep sweep_ditto.yaml
wandb sweep sweep_fedEM.yaml
wandb sweep sweep_pFedMe.yaml
wandb sweep sweep_fedAvg.yaml
wandb sweep sweep_fedOpt.yaml
wandb sweep sweep_fedOpt_FT.yaml
wandb sweep sweep_pFedMe_FT.yaml
wandb sweep sweep_isolated.yaml
wandb sweep sweep_global_train.yaml
wandb sweep sweep_ditto_FT_fedOpt.yaml
wandb sweep sweep_fedAvg_FT_fedOpt.yaml
wandb sweep sweep_fedEM_FT_fedOpt.yaml
wandb sweep sweep_fedEM_FT.yaml
wandb sweep sweep_ditto_FT.yaml


set -e

wandb sweep sweep_fedOpt.yaml
wandb sweep sweep_fedOpt_FT.yaml

wandb sweep sweep_ditto_fedBN.yaml
wandb sweep sweep_ditto_fedBN_fedOpt.yaml
wandb sweep sweep_ditto_FT_fedBN.yaml
wandb sweep sweep_ditto_FT_fedBN_fedOpt.yaml

wandb sweep sweep_fedBN_fedOpt.yaml
wandb sweep sweep_fedBN_FT_fedOpt.yaml

wandb sweep sweep_fedEM_fedBN.yaml
wandb sweep sweep_fedEM_fedBN_fedOpt.yaml
wandb sweep sweep_fedEM_FT_fedBN.yaml
wandb sweep sweep_fedEM_FT_fedBN_fedOpt.yaml

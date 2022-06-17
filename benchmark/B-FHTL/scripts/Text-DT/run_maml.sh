cd ../FederatedScope/federatedscope/

python main.py --cfg contrib/configs/config_maml.yaml --cfg_client contrib/configs/config_client_maml.yaml outdir exp/sts_imdb_squad/maml/

python main.py --cfg contrib/configs/config_ft.yaml --cfg_client contrib/configs/config_client_maml_ft.yaml outdir exp/sts_imdb_squad/maml/ federate.method maml federate.load_from exp/sts_imdb_squad/maml/ckpt/

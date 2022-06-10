cd ../FederatedScope/federatedscope/

python main.py --cfg contrib/configs/config_ft.yaml --cfg_client contrib/configs/config_client_fedbn_ft.yaml outdir exp/sts_imdb_squad/fedbn_ft/ federate.method fedbn federate.load_from exp/sts_imdb_squad/fedbn/ckpt/

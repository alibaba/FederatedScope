R=400
E=twitter_exp
P=twitter
D=0
s=12345
mkdir $E/s$s
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/pfedhpo/$P/pfedhpo.yaml  hpo.pfedhpo.train_fl True hpo.pfedhpo.train_anchor True federate.sample_client_rate 1.0 federate.total_round_num $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/pfedhpo/$P/pfedhpo.yaml  hpo.pfedhpo.train_fl False hpo.pfedhpo.target_fl_total_round $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/pfedhpo/$P/pfedhpo.yaml  hpo.pfedhpo.train_fl True federate.total_round_num $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0
#rm -rf $E/s$s/working/temp_model_round_*
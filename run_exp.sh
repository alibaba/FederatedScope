R=400
E=femnist_exp
P=femnist
D=0
for s in {1..5}
do
mkdir $E/s$s
cp -r $E/working $E/s$s/working
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/$P/pfedhpo_0.yaml  hpo.pfedhpo.train_fl True hpo.pfedhpo.train_anchor True federate.sample_client_rate 1.0 federate.total_round_num $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0;
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/$P/pfedhpo_0.yaml  hpo.pfedhpo.train_fl False hpo.pfedhpo.target_fl_total_round $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0;
CUDA_VISIBLE_DEVICES=$D python federatedscope/main.py --cfg scripts/example_configs/$P/pfedhpo_0.yaml  hpo.pfedhpo.train_fl True federate.total_round_num $R seed $s outdir $E/s$s hpo.working_folder $E/s$s/working device 0;
rm -rf $E/s$s/working/temp_model_round_*;
done

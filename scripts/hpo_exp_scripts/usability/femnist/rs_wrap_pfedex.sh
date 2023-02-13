set -e

# best choice is hpo.fedex.pi_lr 0.01 hpo.fedex.diff False
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 1.0 device 0 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_0 >rs_wrap_pfedex_0.out 2>rs_wrap_pfedex_0.err &
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.1 device 1 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_1 >rs_wrap_pfedex_1.out 2>rs_wrap_pfedex_1.err &
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.01 device 2 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_2 >rs_wrap_pfedex_2.out 2>rs_wrap_pfedex_2.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 1.0 device 3 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_3 >rs_wrap_pfedex_3.out 2>rs_wrap_pfedex_3.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.1 device 4 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_4 >rs_wrap_pfedex_4.out 2>rs_wrap_pfedex_4.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.01 device 5 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_5 >rs_wrap_pfedex_5.out 2>rs_wrap_pfedex_5.err &

# repeat the search procedures 3 times
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12345 hpo.fedex.pi_lr 0.01 device 0 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_0 >rs_wrap_pfedex_0.out 2>rs_wrap_pfedex_0.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12346 hpo.fedex.pi_lr 0.01 device 1 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_1 >rs_wrap_pfedex_1.out 2>rs_wrap_pfedex_1.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12347 hpo.fedex.pi_lr 0.01 device 2 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_2 >rs_wrap_pfedex_2.out 2>rs_wrap_pfedex_2.err &

# find the optimal table idx from rs_wrap_pfedex_?.err
#python federatedscope/autotune/fedex/utils.py --ss_path rs_wrap_pfedex_0/34_tmp_grid_search_space.yaml --log_path rs_wrap_pfedex_0/idx_34_fedex.yaml --pt_path rs_wrap_pfedex_0/idx_34_pfedex.pt --save_path rs_wrap_pfedex_0/clientwise_config.yaml
#python federatedscope/autotune/fedex/utils.py --ss_path rs_wrap_pfedex_1/47_tmp_grid_search_space.yaml --log_path rs_wrap_pfedex_1/idx_47_fedex.yaml --pt_path rs_wrap_pfedex_1/idx_47_pfedex.pt --save_path rs_wrap_pfedex_1/clientwise_config.yaml
#python federatedscope/autotune/fedex/utils.py --ss_path rs_wrap_pfedex_2/77_tmp_grid_search_space.yaml --log_path rs_wrap_pfedex_2/idx_77_fedex.yaml --pt_path rs_wrap_pfedex_2/idx_77_pfedex.pt --save_path rs_wrap_pfedex_2/clientwise_config.yaml

# learn from scratch with optimal hyperparameter configs
#CUDA_VISIBLE_DEVICES="0" python federatedscope/main.py --cfg scripts/example_configs/femnist/avg/learn_from_scratch.yaml --client_cfg rs_wrap_pfedex_0/clientwise_config.yaml expname opt0_lfs_exp device 0 >/dev/null 2>/dev/null &
#CUDA_VISIBLE_DEVICES="1" python federatedscope/main.py --cfg scripts/example_configs/femnist/avg/learn_from_scratch.yaml --client_cfg rs_wrap_pfedex_1/clientwise_config.yaml expname opt1_lfs_exp device 0 >/dev/null 2>/dev/null &
#CUDA_VISIBLE_DEVICES="2" python federatedscope/main.py --cfg scripts/example_configs/femnist/avg/learn_from_scratch.yaml --client_cfg rs_wrap_pfedex_2/clientwise_config.yaml expname opt2_lfs_exp device 0 >/dev/null 2>/dev/null &

# results
# 0.7190350102971462 + 0.8758458370108856 + 0.8615769343924684

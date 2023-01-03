set -e

# best choice is hpo.fedex.pi_lr 0.01 hpo.fedex.diff False
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 1.0 device 0 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_0 >rs_wrap_pfedex_0.out 2>rs_wrap_pfedex_0.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.1 device 1 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_1 >rs_wrap_pfedex_1.out 2>rs_wrap_pfedex_1.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.01 device 2 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_2 >rs_wrap_pfedex_2.out 2>rs_wrap_pfedex_2.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 1.0 device 3 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_3 >rs_wrap_pfedex_3.out 2>rs_wrap_pfedex_3.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.1 device 4 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_4 >rs_wrap_pfedex_4.out 2>rs_wrap_pfedex_4.err &
#python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True hpo.fedex.pi_lr 0.01 device 5 hpo.fedex.diff True hpo.working_folder rs_wrap_pfedex_5 >rs_wrap_pfedex_5.out 2>rs_wrap_pfedex_5.err &

# repeat the search procedures 3 times
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12345 hpo.fedex.pi_lr 0.01 device 0 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_0 >rs_wrap_pfedex_0.out 2>rs_wrap_pfedex_0.err &
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12346 hpo.fedex.pi_lr 0.01 device 1 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_1 >rs_wrap_pfedex_1.out 2>rs_wrap_pfedex_1.err &
python federatedscope/hpo.py --cfg scripts/example_configs/femnist/avg/rs_wrap.yaml hpo.fedex.psn True seed 12347 hpo.fedex.pi_lr 0.01 device 2 hpo.fedex.diff False hpo.working_folder rs_wrap_pfedex_2 >rs_wrap_pfedex_2.out 2>rs_wrap_pfedex_2.err &

# learn from scratch with optimal hyperparameter configs
#python federatedscope/main.py --cfg scripts/example_configs/femnist/avg/learn_from_scratch.yaml --client_cfg rs_wrap_pfedex_2.yaml expname pfedex_exp

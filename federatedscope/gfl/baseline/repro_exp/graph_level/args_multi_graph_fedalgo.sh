# ---------------------------------------------------------------------- #
# FedOpt
# ---------------------------------------------------------------------- #

# mol
bash run_multi_opt.sh 5 mol gcn 0.25 16 &

bash run_multi_opt.sh 7 mol gin 0.25 4 &

bash run_multi_opt.sh 5 mol gat 0.25 16 &

# ---------------------------------------------------------------------- #
# FedProx
# ---------------------------------------------------------------------- #

# mol
bash run_multi_prox.sh 7 mol gcn 0.25 16 &

bash run_multi_prox.sh 5 mol gin 0.01 4 &

bash run_multi_prox.sh 7 mol gat 0.25 16 &



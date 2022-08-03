# ---------------------------------------------------------------------- #
# FedOpt
# ---------------------------------------------------------------------- #

# WN18
bash run_link_level_opt.sh 0 wn18 rel_type gcn 0.25 16 &

bash run_link_level_opt.sh 1 wn18 rel_type sage 0.05 16 &

bash run_link_level_opt.sh 2 wn18 rel_type gat 0.25 16 &

bash run_link_level_opt.sh 3 wn18 rel_type gpr 0.01 16 &

# FB15k-237
bash run_link_level_opt.sh 4 fb15k-237 rel_type gcn 0.25 4 &

bash run_link_level_opt.sh 5 fb15k-237 rel_type sage 0.25 4 &

bash run_link_level_opt.sh 6 fb15k-237 rel_type gat 0.25 4 &

bash run_link_level_opt.sh 7 fb15k-237 rel_type gpr 0.25 1 &


# ---------------------------------------------------------------------- #
# FedProx
# ---------------------------------------------------------------------- #
# WN18
bash run_link_level_prox.sh 7 wn18 rel_type gcn 0.25 16 &

bash run_link_level_prox.sh 6 wn18 rel_type sage 0.05 16 &

bash run_link_level_prox.sh 5 wn18 rel_type gat 0.25 16 &

bash run_link_level_prox.sh 4 wn18 rel_type gpr 0.01 16 &

# FB15k-237
bash run_link_level_prox.sh 3 fb15k-237 rel_type gcn 0.25 4 &

bash run_link_level_prox.sh 2 fb15k-237 rel_type sage 0.25 4 &

bash run_link_level_prox.sh 1 fb15k-237 rel_type gat 0.25 4 &

bash run_link_level_prox.sh 0 fb15k-237 rel_type gpr 0.25 1 &


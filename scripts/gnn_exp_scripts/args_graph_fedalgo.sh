# ---------------------------------------------------------------------- #
# FedOpt
# ---------------------------------------------------------------------- #

# proteins
bash run_graph_level_opt.sh 0 proteins gcn 0.25 4 &

bash run_graph_level_opt.sh 1 proteins gin 0.25 1 &

bash run_graph_level_opt.sh 2 proteins gat 0.25 4 &

# imdb-binary
bash run_graph_level_opt.sh 3 imdb-binary gcn 0.25 16 &

bash run_graph_level_opt.sh 4 imdb-binary gin 0.01 16 &

bash run_graph_level_opt.sh 5 imdb-binary gat 0.25 16 &

# ---------------------------------------------------------------------- #
# FedProx
# ---------------------------------------------------------------------- #

# proteins
bash run_graph_level_prox.sh 6 proteins gcn 0.25 4 &

bash run_graph_level_prox.sh 7 proteins gin 0.25 1 &

bash run_graph_level_prox.sh 1 proteins gat 0.25 4 &

# imdb-binary
bash run_graph_level_prox.sh 2 imdb-binary gcn 0.25 16 &

bash run_graph_level_prox.sh 3 imdb-binary gin 0.01 16 &

bash run_graph_level_prox.sh 4 imdb-binary gat 0.25 16 &



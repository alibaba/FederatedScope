# ---------------------------------------------------------------------- #
# FedOpt
# ---------------------------------------------------------------------- #

# Cora louvain
bash run_node_level_opt.sh 0 cora louvain gcn 0.25 4 &

bash run_node_level_opt.sh 1 cora louvain sage 0.25 16 &

bash run_node_level_opt.sh 2 cora louvain gat 0.25 16 &

bash run_node_level_opt.sh 3 cora louvain gpr 0.25 1 &

# CiteSeer louvain
bash run_node_level_opt.sh 4 citeseer louvain gcn 0.05 1 &

bash run_node_level_opt.sh 5 citeseer louvain sage 0.05 1 &

bash run_node_level_opt.sh 6 citeseer louvain gat 0.01 4 &

bash run_node_level_opt.sh 7 citeseer louvain gpr 0.25 1 &

# PubMed louvain
bash run_node_level_opt.sh 0 pubmed louvain gcn 0.25 16 &

bash run_node_level_opt.sh 4 pubmed louvain sage 0.25 16 &

bash run_node_level_opt.sh 0 pubmed louvain gat 0.25 16 &

bash run_node_level_opt.sh 3 pubmed louvain gpr 0.05 16 &


# Cora random
bash run_node_level_opt.sh 5 cora random gcn 0.25 4 &

bash run_node_level_opt.sh 5 cora random sage 0.25 16 &

bash run_node_level_opt.sh 5 cora random gat 0.25 4 &

bash run_node_level_opt.sh 5 cora random gpr 0.25 1 &

# CiteSeer random
bash run_node_level_opt.sh 2 citeseer random gcn 0.01 4 &

bash run_node_level_opt.sh 2 citeseer random sage 0.01 4 &

bash run_node_level_opt.sh 0 citeseer random gat 0.01 4 &

bash run_node_level_opt.sh 7 citeseer random gpr 0.25 1 &

# PubMed random
bash run_node_level_opt.sh 0 pubmed random gcn 0.25 16 &

bash run_node_level_opt.sh 2 pubmed random sage 0.25 16 &

bash run_node_level_opt.sh 6 pubmed random gat 0.25 16 &

bash run_node_level_opt.sh 7 pubmed random gpr 0.25 16 &



# ---------------------------------------------------------------------- #
# FedProx
# ---------------------------------------------------------------------- #
# Cora louvain
bash run_node_level_prox.sh 0 cora louvain gcn 0.25 4 &

bash run_node_level_prox.sh 1 cora louvain sage 0.25 16 &

bash run_node_level_prox.sh 2 cora louvain gat 0.25 16 &

bash run_node_level_prox.sh 3 cora louvain gpr 0.25 1 &

# CiteSeer louvain
bash run_node_level_prox.sh 0 citeseer louvain gcn 0.05 1 &

bash run_node_level_prox.sh 1 citeseer louvain sage 0.05 1 &

bash run_node_level_prox.sh 2 citeseer louvain gat 0.01 4 &

bash run_node_level_prox.sh 3 citeseer louvain gpr 0.25 1 &

# PubMed louvain
bash run_node_level_prox.sh 4 pubmed louvain gcn 0.25 16 &

bash run_node_level_prox.sh 5 pubmed louvain sage 0.25 16 &

bash run_node_level_prox.sh 6 pubmed louvain gat 0.25 16 &

bash run_node_level_prox.sh 7 pubmed louvain gpr 0.05 16 &

# Cora random
bash run_node_level_prox.sh 4 cora random gcn 0.25 4 &

bash run_node_level_prox.sh 5 cora random sage 0.25 16 &

bash run_node_level_prox.sh 6 cora random gat 0.25 4 &

bash run_node_level_prox.sh 7 cora random gpr 0.25 1 &

# CiteSeer random
bash run_node_level_prox.sh 4 citeseer random gcn 0.01 4 &

bash run_node_level_prox.sh 5 citeseer random sage 0.01 4 &

bash run_node_level_prox.sh 6 citeseer random gat 0.01 4 &

bash run_node_level_prox.sh 7 citeseer random gpr 0.25 1 &

# PubMed random
bash run_node_level_prox.sh 0 pubmed random gcn 0.25 16 &

bash run_node_level_prox.sh 1 pubmed random sage 0.25 16 &

bash run_node_level_prox.sh 2 pubmed random gat 0.25 16 &

bash run_node_level_prox.sh 3 pubmed random gpr 0.25 16 &

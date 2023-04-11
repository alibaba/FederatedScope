set -e

# inconsistent data
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_sage_minibatch_on_dblpnew.yaml federate.total_round_num 20
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_fullbatch_on_dblpnew.yaml federate.total_round_num 20

# standalone
python federatedscope/main.py --cfg federatedscope/example_configs/single_process.yaml
# standalone_local
python federatedscope/main.py --cfg federatedscope/gfl/baseline/local_gnn_node_fullbatch_citation.yaml

# fedavg link_level KG hits@5 evaluation
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_kg.yaml

# fedavg graph_level hiv
python federatedscope/main.py --cfg federatedscope/gfl/baseline/fedavg_gcn_minibatch_on_hiv.yaml

# server in distribute
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_server.yaml
# client in distribute
python federatedscope/main.py --cfg federatedscope/example_configs/distributed_client.yaml

# HPO
python federatedscope/hpo.py --cfg federatedscope/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device 2 data.type cora data.splitter louvain model.type gcn model.out_channels 7 model.hidden 64 seed 123

# femnist & CNN
python federatedscope/main.py --cfg federatedscope/cv/baseline/fedavg_convnet2_on_femnist.yaml >> fedavg_convnet2_on_femnist.out 2>>fedavg_convnet2_on_femnist.err
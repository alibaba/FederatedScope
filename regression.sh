set -e

# inconsistent data
python flpackage/main.py --cfg flpackage/gfl/baseline/fedavg_sage_minibatch_on_dblpnew.yaml federate.total_round_num 20
python flpackage/main.py --cfg flpackage/gfl/baseline/fedavg_gcn_fullbatch_on_dblpnew.yaml federate.total_round_num 20

# standalone
python flpackage/main.py --cfg flpackage/example_configs/single_process.yaml
# standalone_local
python flpackage/main.py --cfg flpackage/gfl/baseline/local_gnn_node_fullbatch_citation.yaml

# fedavg link_level KG hits@5 evaluation
python flpackage/main.py --cfg flpackage/gfl/baseline/fedavg_gcn_minibatch_on_kg.yaml

# fedavg graph_level hiv
python flpackage/main.py --cfg flpackage/gfl/baseline/fedavg_gcn_minibatch_on_hiv.yaml

# server in distribute
python flpackage/main.py --cfg flpackage/example_configs/distributed_server.yaml
# client in distribute
python flpackage/main.py --cfg flpackage/example_configs/distributed_client.yaml

# HPO
python flpackage/hpo.py --cfg flpackage/gfl/baseline/fedavg_gnn_node_fullbatch_citation.yaml device 2 data.type cora data.splitter louvain model.type gcn model.out_channels 7 model.hidden 64 seed 123

# femnist & CNN
python flpackage/main.py --cfg flpackage/cv/baseline/fedavg_convnet2_on_femnist.yaml >> fedavg_convnet2_on_femnist.out 2>>fedavg_convnet2_on_femnist.err
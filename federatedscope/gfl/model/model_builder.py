from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.gfl.model.gcn import GCN_Net
from federatedscope.gfl.model.sage import SAGE_Net
from federatedscope.gfl.model.gat import GAT_Net
from federatedscope.gfl.model.gin import GIN_Net
from federatedscope.gfl.model.gpr import GPR_Net
from federatedscope.gfl.model.link_level import GNN_Net_Link
from federatedscope.gfl.model.graph_level import GNN_Net_Graph
from federatedscope.gfl.model.mpnn import MPNNs2s


def get_gnn(model_config, input_shape):

    x_shape, num_label, num_edge_features = input_shape
    if model_config.task.startswith('node'):
        if model_config.type == 'gcn':
            # assume `data` is a dict where key is the client index,
            # and value is a PyG object
            model = GCN_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'sage':
            model = SAGE_Net(x_shape[-1],
                             model_config.out_channels,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout)
        elif model_config.type == 'gat':
            model = GAT_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gin':
            model = GIN_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            max_depth=model_config.layer,
                            dropout=model_config.dropout)
        elif model_config.type == 'gpr':
            model = GPR_Net(x_shape[-1],
                            model_config.out_channels,
                            hidden=model_config.hidden,
                            K=model_config.layer,
                            dropout=model_config.dropout)
        else:
            raise ValueError('not recognized gnn model {}'.format(
                model_config.type))

    elif model_config.task.startswith('link'):
        model = GNN_Net_Link(x_shape[-1],
                             model_config.out_channels,
                             hidden=model_config.hidden,
                             max_depth=model_config.layer,
                             dropout=model_config.dropout,
                             gnn=model_config.type)
    elif model_config.task.startswith('graph'):
        if model_config.type == 'mpnn':
            model = MPNNs2s(in_channels=x_shape[-1],
                            out_channels=model_config.out_channels,
                            num_nn=num_edge_features,
                            hidden=model_config.hidden)
        else:
            model = GNN_Net_Graph(x_shape[-1],
                                  max(model_config.out_channels, num_label),
                                  hidden=model_config.hidden,
                                  max_depth=model_config.layer,
                                  dropout=model_config.dropout,
                                  gnn=model_config.type,
                                  pooling=model_config.graph_pooling)
    else:
        raise ValueError('not recognized data task {}'.format(
            model_config.task))
    return model

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from federatedscope.gfkd.model.graph2D_model import GNN_Net
from federatedscope.gfkd.model.graph3D_model import DimeNetPlusPlus_for_QM7b
from federatedscope.gfkd.model.SMILES_model import SMILESTransformer


def get_gnn(model_config, input_shape):

    x_shape, num_label, num_edge_features = input_shape
    if not num_label:
        num_label = 0
    if model_config.type == 'SMILESTransformer':
        # assume `data` is a dict where key is the client index,
        # and value is a PyG object
        model = SMILESTransformer(ntoken=415, 
                                ninp=128, 
                                nhead=8, 
                                nhid=model_config.hidden, 
                                nlayers=model_config.layer,
                                dropout=model_config.dropout)
    elif model_config.type == 'GNN_Net':
        model = GNN_Net(x_shape[-1],
                                max(model_config.out_channels, num_label),
                                hidden=model_config.hidden,
                                max_depth=model_config.layer,
                                dropout=model_config.dropout,
                                gnn=model_config.type,
                                pooling=model_config.graph_pooling)
    elif model_config.type == 'DimeNetPlusPlus':
        model = DimeNetPlusPlus_for_QM7b(hidden_channels=model_config.hidden,
                        out_channels=model_config.out_channels,
                        num_blocks=model_config.num_blocks,
                        int_emb_size=model_config.int_emb_size,
                        basis_emb_size=model_config.basis_emb_size,
                        out_emb_channels=model_config.out_emb_channels)
    else:
        raise ValueError('not recognized model {}'.format(
            model_config.type))

    return model

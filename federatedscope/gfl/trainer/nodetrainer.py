import torch

from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler

from federatedscope.register import register_trainer
from federatedscope.core.trainers.trainer import GeneralTorchTrainer
from federatedscope.core.auxiliaries.ReIterator import ReIterator


class NodeFullBatchTrainer(GeneralTorchTrainer):
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

        """
        init_dict = dict()
        if isinstance(data, Data):
            for mode in ["train", "val", "test"]:
                init_dict["{}_loader".format(mode)] = PyGDataLoader([data])
                init_dict["{}_data".format(mode)] = None
                # For node-level task dataloader contains one graph
                init_dict["num_{}_data".format(mode)] = 1

        else:
            raise TypeError("Type of data should be PyG data.")
        return init_dict

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)
        pred = ctx.model(batch)[batch['{}_mask'.format(ctx.cur_data_split)]]
        label = batch.y[batch['{}_mask'.format(ctx.cur_data_split)]]
        ctx.batch_size = torch.sum(ctx.data_batch['{}_mask'.format(
            ctx.cur_data_split)]).item()

        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred


class NodeMiniBatchTrainer(GeneralTorchTrainer):
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different modes

        """
        init_dict = dict()
        if isinstance(data, dict):
            for mode in ["train", "val", "test"]:
                init_dict["{}_data".format(mode)] = None
                init_dict["{}_loader".format(mode)] = None
                init_dict["num_{}_data".format(mode)] = 0
                if data.get(mode, None) is not None:
                    if isinstance(
                            data.get(mode), NeighborSampler) or isinstance(
                                data.get(mode), GraphSAINTRandomWalkSampler):
                        if mode == 'train':
                            init_dict["{}_loader".format(mode)] = data.get(
                                mode)
                            init_dict["num_{}_data".format(mode)] = len(
                                data.get(mode).dataset)
                        else:
                            # We need to pass Full Dataloader to model
                            init_dict["{}_loader".format(mode)] = [
                                data.get(mode)
                            ]
                            init_dict["num_{}_data".format(
                                mode)] = self.cfg.data.batch_size
                    else:
                        raise TypeError("Type {} is not supported.".format(
                            type(data.get(mode))))
        else:
            raise TypeError("Type of data should be dict.")
        return init_dict

    def _hook_on_epoch_start(self, ctx):
        # TODO: blind torch
        if not isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)),
                          ReIterator):
            if isinstance(ctx.get("{}_loader".format(ctx.cur_data_split)),
                          NeighborSampler):
                self.is_NeighborSampler = True
                ctx.data['data'].x = ctx.data['data'].x.to(ctx.device)
                ctx.data['data'].y = ctx.data['data'].y.to(ctx.device)
            else:
                self.is_NeighborSampler = False
            setattr(
                ctx, "{}_loader".format(ctx.cur_data_split),
                ReIterator(ctx.get("{}_loader".format(ctx.cur_data_split))))

    def _hook_on_batch_forward(self, ctx):
        if ctx.cur_data_split == 'train':
            # For training
            if self.is_NeighborSampler:
                # For NeighborSamper
                batch_size, n_id, adjs = ctx.data_batch
                adjs = [adj.to(ctx.device) for adj in adjs]
                pred = ctx.model(ctx.data['data'].x[n_id], adjs=adjs)
                label = ctx.data['data'].y[n_id[:batch_size]]
                ctx.batch_size, _, _ = ctx.data_batch
            else:
                # For GraphSAINTRandomWalkSampler or PyGDataLoader
                batch = ctx.data_batch.to(ctx.device)
                pred = ctx.model(batch.x,
                                 batch.edge_index)[batch['{}_mask'.format(
                                     ctx.cur_data_split)]]
                label = batch.y[batch['{}_mask'.format(ctx.cur_data_split)]]
                ctx.batch_size = torch.sum(ctx.data_batch['train_mask']).item()
        else:
            # For inference
            subgraph_loader = ctx.data_batch
            mask = ctx.data['data']['{}_mask'.format(ctx.cur_data_split)]
            pred = ctx.model.inference(ctx.data['data'].x, subgraph_loader,
                                       ctx.device)[mask]
            label = ctx.data['data'].y[mask]
            ctx.batch_size = torch.sum(ctx.data['data']['{}_mask'.format(
                ctx.cur_data_split)]).item()

        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred


def call_node_level_trainer(trainer_type):
    if trainer_type == 'nodefullbatch_trainer':
        trainer_builder = NodeFullBatchTrainer
    elif trainer_type == 'nodeminibatch_trainer':
        trainer_builder = NodeMiniBatchTrainer
    else:
        raise ValueError

    return trainer_builder


register_trainer('nodefullbatch_trainer', call_node_level_trainer)
register_trainer('nodeminibatch_trainer', call_node_level_trainer)

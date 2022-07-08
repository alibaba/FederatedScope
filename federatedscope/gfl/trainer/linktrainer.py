import torch

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler

from federatedscope.core.monitors import Monitor
from federatedscope.register import register_trainer
from federatedscope.core.trainers import GeneralTorchTrainer

import logging

logger = logging.getLogger(__name__)

MODE2MASK = {
    'train': 'train_edge_mask',
    'val': 'valid_edge_mask',
    'test': 'test_edge_mask'
}


class LinkFullBatchTrainer(GeneralTorchTrainer):
    def register_default_hooks_eval(self):
        super().register_default_hooks_eval()
        self.register_hook_in_eval(
            new_hook=self._hook_on_epoch_start_data2device,
            trigger='on_fit_start',
            insert_pos=-1)

    def register_default_hooks_train(self):
        super().register_default_hooks_train()
        self.register_hook_in_train(
            new_hook=self._hook_on_epoch_start_data2device,
            trigger='on_fit_start',
            insert_pos=-1)

    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes

        """
        init_dict = dict()
        if isinstance(data, Data):
            for mode in ["train", "val", "test"]:
                edges = data.edge_index.T[data[MODE2MASK[mode]]]
                # Use an index loader
                index_loader = DataLoader(range(edges.size(0)),
                                          self.cfg.data.batch_size,
                                          shuffle=self.cfg.data.shuffle
                                          if mode == 'train' else False,
                                          drop_last=self.cfg.data.drop_last
                                          if mode == 'train' else False)
                init_dict["{}_loader".format(mode)] = index_loader
                init_dict["num_{}_data".format(mode)] = edges.size(0)
                init_dict["{}_data".format(mode)] = None
        else:
            raise TypeError("Type of data should be PyG data.")
        return init_dict

    def _hook_on_epoch_start_data2device(self, ctx):
        ctx.data = ctx.data.to(ctx.device)
        # For handling different dict key
        if "input_edge_index" in ctx.data:
            ctx.input_edge_index = ctx.data.input_edge_index
        else:
            ctx.input_edge_index = ctx.data.edge_index.T[
                ctx.data.train_edge_mask].T

    def _hook_on_batch_forward(self, ctx):
        data = ctx.data
        perm = ctx.data_batch
        mask = ctx.data[MODE2MASK[ctx.cur_data_split]]
        edges = data.edge_index.T[mask]
        if ctx.cur_data_split in ['train', 'val']:
            h = ctx.model((data.x, ctx.input_edge_index))
        else:
            h = ctx.model((data.x, data.edge_index))
        pred = ctx.model.link_predictor(h, edges[perm].T)
        label = data.edge_type[mask][perm]  # edge_type is y

        ctx.loss_batch = ctx.criterion(pred, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

    def _hook_on_batch_forward_flop_count(self, ctx):
        if not isinstance(self.ctx.monitor, Monitor):
            logger.warning(
                f"The trainer {type(self)} does contain a valid monitor, "
                f"this may be caused by initializing trainer subclasses "
                f"without passing a valid monitor instance."
                f"Plz check whether this is you want.")
            return

        if self.cfg.eval.count_flops and self.ctx.monitor.flops_per_sample \
                == 0:
            # calculate the flops_per_sample
            try:
                data = ctx.data
                from fvcore.nn import FlopCountAnalysis
                if ctx.cur_data_split in ['train', 'val']:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, (data.x, ctx.input_edge_index)).total()
                else:
                    flops_one_batch = FlopCountAnalysis(
                        ctx.model, (data.x, data.edge_index)).total()
                if self.model_nums > 1 and ctx.mirrored_models:
                    flops_one_batch *= self.model_nums
                    logger.warning(
                        "the flops_per_batch is multiplied by "
                        "internal model nums as self.mirrored_models=True."
                        "if this is not the case you want, "
                        "please customize the count hook.")
                self.ctx.monitor.track_avg_flops(flops_one_batch,
                                                 ctx.batch_size)
            except:
                logger.warning(
                    "current flop count implementation is for general "
                    "NodeFullBatchTrainer case: "
                    "1) the ctx.model takes the "
                    "tuple (data.x, data.edge_index) or tuple (data.x, "
                    "ctx.input_edge_index) as input."
                    "Please check the forward format or implement your own "
                    "flop_count function")
                # warning at the first failure
                self.ctx.monitor.flops_per_sample = -1


class LinkMiniBatchTrainer(GeneralTorchTrainer):
    """
        # Support GraphSAGE with GraphSAINTRandomWalkSampler in train ONLY!
    """
    def parse_data(self, data):
        """Populate "{}_data", "{}_loader" and "num_{}_data" for different
        modes

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

    def _hook_on_batch_forward(self, ctx):
        if ctx.cur_data_split == 'train':
            batch = ctx.data_batch.to(ctx.device)
            mask = batch[MODE2MASK[ctx.cur_data_split]]
            edges = batch.edge_index.T[mask].T
            h = ctx.model((batch.x, edges))
            pred = ctx.model.link_predictor(h, edges)
            label = batch.edge_type[mask]
            ctx.batch_size = torch.sum(
                ctx.data_batch[MODE2MASK[ctx.cur_data_split]]).item()
        else:
            # For inference
            mask = ctx.data['data'][MODE2MASK[ctx.cur_data_split]]
            subgraph_loader = ctx.data_batch
            h = ctx.model.gnn.inference(ctx.data['data'].x, subgraph_loader,
                                        ctx.device).to(ctx.device)
            edges = ctx.data['data'].edge_index.T[mask].to(ctx.device)
            pred = []

            for perm in DataLoader(range(edges.size(0)),
                                   self.cfg.data.batch_size):
                edge = edges[perm].T
                pred += [ctx.model.link_predictor(h, edge).squeeze()]
            pred = torch.cat(pred, dim=0)
            label = ctx.data['data'].edge_type[mask].to(ctx.device)
            ctx.batch_size = torch.sum(
                ctx.data['data'][MODE2MASK[ctx.cur_data_split]]).item()

        ctx.loss_batch = ctx.criterion(pred, label)
        ctx.y_true = label
        ctx.y_prob = pred


def call_link_level_trainer(trainer_type):
    if trainer_type == 'linkfullbatch_trainer':
        trainer_builder = LinkFullBatchTrainer
    elif trainer_type == 'linkminibatch_trainer':
        trainer_builder = LinkMiniBatchTrainer
    else:
        raise ValueError

    return trainer_builder


register_trainer('linkfullbatch_trainer', call_link_level_trainer)
register_trainer('linkminibatch_trainer', call_link_level_trainer)

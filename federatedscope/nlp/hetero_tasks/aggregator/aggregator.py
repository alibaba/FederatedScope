import os
import re
import copy
import random
import torch
import numpy as np
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.configs.config import global_cfg

logger = logging.getLogger(__name__)


class ATCAggregator(ClientsAvgAggregator):
    def __init__(self, model=None, config=None, device='cpu'):
        super().__init__(model=model, config=config, device=device)
        self.client_num = config.federate.client_num
        self.task = config.model.task

        self.pretrain_tasks = config.model.pretrain_tasks
        self.num_agg_groups = config.aggregator.num_agg_groups
        self.num_agg_topk = config.aggregator.num_agg_topk
        self.inside_weight = config.aggregator.inside_weight
        self.outside_weight = config.aggregator.outside_weight
        self.models = []
        self.neighbors = {}
        self.client_id2group = [None for _ in range(self.client_num)]
        self.client_id2topk = [[] for _ in range(self.client_num)]
        self.client_id2all = [[] for _ in range(self.client_num)]

        self.use_contrastive_loss = config.model.use_contrastive_loss
        if self.use_contrastive_loss:
            self.contrast_monitor = None

    def update_models(self, models):
        self.models = models

    def update_neighbors(self, neighbors):
        self.neighbors = neighbors

    def update_contrast_monitor(self, contrast_monitor):
        self.contrast_monitor = contrast_monitor

    def aggregate(self, agg_info):
        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
                'recover_fun' in agg_info and global_cfg.federate.use_ss) \
            else None
        avg_models, tasks = self._para_weighted_avg(models,
                                                    recover_fun=recover_fun)
        return avg_models, tasks

    def update(self, model_parameters):
        for i, param in enumerate(model_parameters):
            self.models[i].load_state_dict(param, strict=False)

    def save_model(self, path, cur_round=-1):
        assert self.models is not None

        path = os.path.join(path, 'global')
        os.makedirs(path, exist_ok=True)
        neighbor_ids = sorted(list(self.neighbors.keys()))
        for i, model in enumerate(self.models):
            ckpt = {'cur_round': cur_round, 'model': model.state_dict()}
            torch.save(
                ckpt,
                os.path.join(path,
                             'global_model_{}.pt'.format(neighbor_ids[i])))

    def load_model(self, path):
        if getattr(self, 'models', None):
            round = None
            global_dir = os.path.join(path, 'global')
            client_dir = os.path.join(path, 'client')
            neighbor_ids = sorted([
                int(re.search(r'model_(\d+).pt', x).groups()[0])
                for x in os.listdir(global_dir)
            ])
            assert len(neighbor_ids) == len(self.models)

            for i, model in enumerate(self.models):
                cur_global_path = os.path.join(
                    global_dir, 'global_model_{}.pt'.format(neighbor_ids[i]))
                cur_client_path = os.path.join(
                    client_dir, 'client_model_{}.pt'.format(neighbor_ids[i]))
                if os.path.exists(cur_global_path):
                    model_ckpt = model.state_dict()
                    logger.info(
                        'Loading model from \'{}\''.format(cur_global_path))
                    global_ckpt = torch.load(cur_global_path,
                                             map_location=self.device)
                    model_ckpt.update(global_ckpt['model'])
                    if os.path.exists(cur_client_path):
                        logger.info('Updating model from \'{}\''.format(
                            cur_client_path))
                        client_ckpt = torch.load(cur_client_path,
                                                 map_location=self.device)
                        model_ckpt.update(client_ckpt['model'])
                    self.models[i].load_state_dict(model_ckpt)
                    round = global_ckpt['cur_round']
                else:
                    raise ValueError(
                        "The file {} does NOT exist".format(cur_global_path))

            return round

    def _compute_client_groups(self, models):
        tasks = [None for _ in range(self.client_num)]

        if self.task == 'pretrain':
            grads = torch.stack([
                torch.cat([g.view(-1) for g in model['model_grads'].values()])
                for model in models
            ])
            clustering = AgglomerativeClustering(
                n_clusters=self.num_agg_groups,
                affinity='cosine',
                linkage='average').fit(grads)
            self.client_id2group = clustering.labels_
            task_id = random.randint(0, len(self.pretrain_tasks) - 1)
            tasks = [
                self.pretrain_tasks[task_id] for _ in range(self.client_num)
            ]
        else:
            grads = torch.stack([
                torch.cat([g.view(-1) for g in model['model_grads'].values()])
                for model in models
            ])
            distances = cosine_distances(grads, grads)
            self.client_id2topk = [
                dis[:k].tolist() for dis, k in zip(
                    np.argsort(distances, axis=-1), self.num_agg_topk)
            ]
            self.client_id2all = np.argsort(distances, axis=-1).tolist()

        return tasks

    def _avg_params(self, models, client_adj_norm):
        avg_model = copy.deepcopy([{
            n: p
            for n, p in model.state_dict().items()
            if n in models[0]['model_grads']
        } for model in self.models])
        model_grads = copy.deepcopy([model['model_grads'] for model in models])
        avg_grads = copy.deepcopy(model_grads)
        for k in avg_model[0]:
            for i in range(len(avg_model)):
                for j in range(len(avg_model)):
                    weight = client_adj_norm[i][j]
                    local_grad = model_grads[j][k].float()
                    if j == 0:
                        avg_grads[i][k] = local_grad * weight
                    else:
                        avg_grads[i][k] += local_grad * weight
                avg_model[i][k] = avg_model[i][k].float() + avg_grads[i][k]

        return avg_model

    def _para_weighted_avg(self, models, recover_fun=None):
        tasks = [None for _ in range(self.client_num)]
        if self.cfg.federate.method in ['local', 'global']:
            model_params = {
                'model_para': [model['model_para'] for model in models]
            }
            return model_params, tasks

        if self.task == 'pretrain':
            # generate self.client_id2group and param weight matrix
            tasks = self._compute_client_groups(models)
            group_id2client = {k: [] for k in range(self.num_agg_groups)}
            for gid in range(self.num_agg_groups):
                for cid in range(self.client_num):
                    if self.client_id2group[cid] == gid:
                        group_id2client[gid].append(cid)
            logger.info('group_id2client: {}'.format({
                k + 1: [x + 1 for x in v]
                for k, v in group_id2client.items()
            }))

            client_adj = torch.zeros(self.client_num, self.client_num)
            for i in range(self.client_num):
                for j in range(self.client_num):
                    if self.client_id2group[i] == self.client_id2group[j]:
                        client_adj[i][j] = models[j]['sample_size'] * \
                                           self.inside_weight
                    else:
                        client_adj[i][j] = models[j]['sample_size'] * \
                                           self.outside_weight
            client_adj_norm = client_adj / client_adj.sum(dim=-1, keepdim=True)

            # aggregate model params
            if not self.use_contrastive_loss:
                model_params = {
                    'model_para': self._avg_params(models, client_adj_norm),
                }
            else:
                model_params = {
                    'model_para': self._avg_params(models, client_adj_norm),
                    'contrast_monitor': self.contrast_monitor
                }

        else:
            if not self.use_contrastive_loss:
                # generate self.client_id2topk and param weight matrix
                tasks = self._compute_client_groups(models)
                logger.info('client_id2topk: {}'.format({
                    k + 1: [x + 1 for x in v] if v else v
                    for k, v in enumerate(self.client_id2topk)
                }))

                client_adj = torch.zeros(self.client_num, self.client_num)
                for i in range(self.client_num):
                    for j in range(self.client_num):
                        if j in self.client_id2topk[i]:
                            client_adj[i][j] = models[j]['sample_size'] * \
                                               self.inside_weight
                        else:
                            client_adj[i][j] = models[j]['sample_size'] * \
                                               self.outside_weight
                client_adj_norm = client_adj / client_adj.sum(dim=-1,
                                                              keepdim=True)

                # aggregate model params
                model_params = {
                    'model_para': self._avg_params(models, client_adj_norm)
                }

            else:
                contrast_stat = models[0]['contrast_monitor'].stat
                for model in models:
                    assert model['contrast_monitor'].stat == contrast_stat
                self.contrast_monitor.update_stat(contrast_stat)
                model_params = None

                if contrast_stat == 2:
                    dec_hidden = [
                        model['contrast_monitor'].dec_hidden
                        for model in models
                    ]
                    dec_out = [
                        model['contrast_monitor'].dec_out for model in models
                    ]
                    dec_hidden = {k + 1: v for k, v in enumerate(dec_hidden)}
                    dec_out = {k + 1: v for k, v in enumerate(dec_out)}
                    all_group_ids = {
                        k + 1: [x + 1 for x in v]
                        for k, v in enumerate(self.client_id2all)
                    }
                    topk_group_ids = {
                        k + 1: [x + 1 for x in v]
                        for k, v in enumerate(self.client_id2topk)
                    }
                    self.contrast_monitor.update_dec_hidden(dec_hidden)
                    self.contrast_monitor.update_dec_out(dec_out)
                    self.contrast_monitor.update_all_group_ids(all_group_ids)
                    self.contrast_monitor.update_topk_group_ids(topk_group_ids)

                elif contrast_stat == 3:
                    # generate self.client_id2topk and param weight matrix
                    tasks = self._compute_client_groups(models)
                    logger.info('client_id2all (n_topk={}): {}'.format(
                        self.num_agg_topk, {
                            k + 1: [x + 1 for x in v]
                            for k, v in enumerate(self.client_id2all)
                        }))

                    client_adj = torch.zeros(self.client_num, self.client_num)
                    for i in range(self.client_num):
                        for j in range(self.client_num):
                            if j in self.client_id2topk[i]:
                                client_adj[i][j] = models[j]['sample_size'] * \
                                                   self.inside_weight
                            else:
                                client_adj[i][j] = models[j]['sample_size'] * \
                                                   self.outside_weight
                    client_adj_norm = client_adj / client_adj.sum(dim=-1,
                                                                  keepdim=True)

                    # aggregate model params
                    model_params = {
                        'model_para': self._avg_params(models, client_adj_norm)
                    }

        return model_params, tasks

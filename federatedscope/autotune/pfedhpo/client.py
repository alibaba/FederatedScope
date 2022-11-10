import os
import logging
import copy

import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from federatedscope.core.message import Message
from federatedscope.core.workers import Client
from federatedscope.autotune.pfedhpo.utils import *

logger = logging.getLogger(__name__)


class pFedHPOClient(Client):

    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):

        super(pFedHPOClient, self).__init__(ID, server_id, state, config,
            data, model, device, strategy, is_unseen_client, *args, **kwargs)

        if self._cfg.hpo.pfedhpo.train_fl and self._cfg.hpo.pfedhpo.train_anchor:

            # feats = []
            # _model = torchvision.models.resnet18(pretrained=True)
            # feat_extractor = create_feature_extractor(_model, {'avgpool': 'feat'})
            # for _, d in enumerate(data['train']):
            #     _d = d[0]
            #     if len(_d.shape) == 4:
            #         if _d.shape[0] == 1:
            #             _d = torch.cat([_d]*2, dim=0)
            #         if _d.shape[1] == 1:
            #             _d = torch.cat([_d]*3, dim=1)
            #         if _d.shape[-1] < 32:
            #             _d = torch.nn.functional.interpolate(_d, size=32, mode='nearest')
            #         try:
            #             out = feat_extractor(_d)
            #         except:
            #             import pdb;pdb.set_trace()
            #         feats.append(out['feat'])
            #
            #     else:
            #         import pdb;pdb.set_trace()
            #         raise NotImplementedError
            # feats = torch.cat(feats).mean(0).squeeze()


            mmd_type = 'sphere'
            d_enc = 32*32*3
            d_rff = 10
            rff_sigma = [127,]
            rff_sigma = [float(sig) for sig in rff_sigma]
            embs = []
            for sig in rff_sigma:
                if mmd_type == 'sphere':
                    w_freq = weights_sphere(d_rff, d_enc, sig, device, seed=1234)
                else:
                    w_freq = weights_rahimi_recht(d_rff, d_enc, sig, device, seed=1234)

                emb = noisy_dataset_embedding(
                    data['train'], w_freq, d_rff, device,
                    n_labels=10, noise_factor=1.0,
                    mmd_type=mmd_type, sum_frequency=25)
                embs.append(emb)
            feats = torch.cat(embs).reshape(-1)


            torch.save(feats, os.path.join(self._cfg.hpo.working_folder, 'client_%d_encoding.pt' % self.ID))

    def _apply_hyperparams(self, hyperparams):
        """Apply the given hyperparameters
        Arguments:
            hyperparams (dict): keys are hyperparameter names \
                and values are specific choices.
        """

        cmd_args = []
        for k, v in hyperparams.items():
            cmd_args.append(k)
            cmd_args.append(v)

        self._cfg.defrost()
        self._cfg.merge_from_list(cmd_args)
        self._cfg.freeze(inform=False)

        self.trainer.ctx.setup_vars()

    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        model_params, hyperparams = content["model_param"], content["hyper_param"]
        attempt = {
            'Role': 'Client #{:d}'.format(self.ID),
            'Hyperparams': hyperparams
        }
        logger.info('-'*30)
        logger.info(attempt)

        if hyperparams is not None:
            self._apply_hyperparams(hyperparams)

        self.trainer.update(model_params)

        # self.model.load_state_dict(content)
        self.state = round
        sample_size, model_para_all, results = self.trainer.train()

        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{}'.format(self.ID),
                                          return_raw=True))

        content = (sample_size, model_para_all, results)
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        if message.content is not None:
            model_params = message.content["model_param"]
            self.trainer.update(model_params)
        if self._cfg.finetune.before_eval:
            self.trainer.finetune()
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(target_data_split_name=split)
            for key in eval_metrics:

                if self._cfg.federate.mode == 'distributed':
                    logger.info('Client #{:d}: (Evaluation ({:s} set) at '
                                'Round #{:d}) {:s} is {:.6f}'.format(
                                    self.ID, split, self.state, key,
                                    eval_metrics[key]))
                metrics.update(**eval_metrics)
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))

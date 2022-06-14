import os
import os.path as osp
import sys

DEV_MODE = True  # simplify the federatedscope re-setup everytime we change the source codes of federatedscope
if DEV_MODE:
    file_dir = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(file_dir)

import copy
from transformers.models.bert import BertTokenizerFast
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.core.fed_runner import FedRunner

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def extend_init_cfg(cfg):
    cfg.federate.load_from = None

    cfg.model.bert_type = None
    cfg.model.dec_d_ffn = None
    cfg.model.dec_dropout_prob = None
    cfg.model.num_dec_layers = None
    cfg.model.num_labels = CN()
    cfg.model.num_labels.sts = None
    cfg.model.num_labels.imdb = None
    cfg.model.num_labels.squad = None
    cfg.model.label_smoothing = None
    cfg.model.maml = None

    cfg.eval.n_best_size = None
    cfg.eval.max_answer_len = None
    cfg.eval.null_score_diff_threshold = None

    cfg.data.dir = CN()
    cfg.data.dir.sts = None
    cfg.data.dir.imdb = None
    cfg.data.dir.squad = None
    cfg.data.max_seq_len = CN()
    cfg.data.max_seq_len.sts = None
    cfg.data.max_seq_len.imdb = None
    cfg.data.max_seq_len.squad = None
    cfg.data.max_tgt_len = None
    cfg.data.max_query_len = None
    cfg.data.trunc_stride = None
    cfg.data.all_batch_size = CN()
    cfg.data.all_batch_size.sts = None
    cfg.data.all_batch_size.imdb = None
    cfg.data.all_batch_size.squad = None
    cfg.data.cache_dir = None

    cfg.scheduler = CN()
    cfg.scheduler.type = None
    cfg.scheduler.warmup_ratio = None

    cfg.trainer.disp_freq = None
    cfg.trainer.val_freq = None
    cfg.trainer.grad_accum_count = None
    cfg.trainer.train_steps = None
    cfg.trainer.test_only = None

    cfg.maml = CN()
    cfg.maml.inner_lr = None

    return cfg


def extend_cfg_client(init_cfg, cfg_client):
    num_clients = len([k for k in cfg_client.keys() if k.startswith('client')])
    for i in range(1, num_clients + 1):
        cfg = cfg_client['client_{}'.format(i)]
        task = cfg.data.type
        cfg.data.batch_size = init_cfg.data.all_batch_size[task]

    with open(osp.join(init_cfg.outdir, 'config_client.yaml'), 'w') as outfile:
        from contextlib import redirect_stdout
        with redirect_stdout(outfile):
            tmp_cfg = copy.deepcopy(cfg_client)
            tmp_cfg.cfg_check_funcs = []
            print(tmp_cfg.dump())

    return cfg_client


def redirect_cfg_dir(cfg):
    if cfg.federate.save_to:
        cfg.federate.save_to = osp.join(cfg.outdir, cfg.federate.save_to)
        os.makedirs(cfg.federate.save_to, exist_ok=True)
    return cfg


if __name__ == '__main__':
    init_cfg = extend_init_cfg(global_cfg.clone())
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    update_logger(init_cfg)
    init_cfg = redirect_cfg_dir(init_cfg)
    setup_seed(init_cfg.seed)

    # set up tokenizer
    bos_token, eos_token, eoq_token = '[unused0]', '[unused1]', '[unused2]'
    tokenizer = BertTokenizerFast.from_pretrained(
        init_cfg.model.bert_type,
        additional_special_tokens=[bos_token, eos_token, eoq_token],
        skip_special_tokens=True,
    )
    data, modified_cfg = get_data(config=init_cfg.clone(), tokenizer=tokenizer)
    init_cfg.merge_from_other_cfg(modified_cfg)
    init_cfg.freeze()

    # allow different settings for different clients
    cfg_client = CN.load_cfg(open(args.cfg_client, 'r'))
    cfg_client = extend_cfg_client(init_cfg, cfg_client)

    runner = FedRunner(data=data,
                       tokenizer=tokenizer,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       config_client=cfg_client)
    _ = runner.run()

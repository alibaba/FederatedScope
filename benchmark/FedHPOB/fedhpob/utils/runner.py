from federatedscope.core.cmd_args import parse_args
from fedhpob.config import fhb_cfg, add_configs
from fedhpob.optimizers import run_dehb, run_hpbandster, run_optuna, \
    run_smac, run_grid_search


def run(cfg):
    if cfg.optimizer.type in ['de', 'dehb']:
        results = run_dehb(cfg)
    elif cfg.optimizer.type in ['rs', 'bo_kde', 'hb', 'bohb']:
        results = run_hpbandster(cfg)
    elif cfg.optimizer.type in ['tpe_md', 'tpe_hb']:
        results = run_optuna(cfg)
    elif cfg.optimizer.type in ['bo_gp', 'bo_rf']:
        results = run_smac(cfg)
    elif cfg.optimizer.type in ['grid_search']:
        results = run_grid_search(cfg)
    else:
        raise NotImplementedError
    return results


def main():
    init_cfg = fhb_cfg.clone()
    args = parse_args()
    init_cfg.merge_from_file(args.cfg_file)
    init_cfg.merge_from_list(args.opts)
    add_configs(init_cfg)
    run(cfg=init_cfg)


if __name__ == '__main__':
    main()

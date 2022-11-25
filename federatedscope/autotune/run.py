def run_scheduler(scheduler, cfg, client_cfgs=None):
    if cfg.hpo.scheduler in ['sha', 'wrap_sha']:
        _ = scheduler.optimize()
    elif cfg.hpo.scheduler in [
            'rs', 'bo_kde', 'hb', 'bohb', 'wrap_rs', 'wrap_bo_kde', 'wrap_hb',
            'wrap_bohb'
    ]:
        from federatedscope.autotune.hpbandster import run_hpbandster
        run_hpbandster(cfg, scheduler, client_cfgs)
    elif cfg.hpo.scheduler in ['bo_gp', 'bo_rf', 'wrap_bo_gp', 'wrap_bo_rf']:
        from federatedscope.autotune.smac import run_smac
        run_smac(cfg, scheduler, client_cfgs)
    else:
        raise ValueError(f'No scheduler named {cfg.hpo.scheduler}')

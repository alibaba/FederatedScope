import os.path as osp
import copy


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

import ConfigSpace as CS
from federatedscope.core.configs.config import CN
from fedhpob.benchmarks import TabularBenchmark
from fedhpob.benchmarks import RawBenchmark
from fedhpob.benchmarks import SurrogateBenchmark

fhb_cfg = CN()


def get_cs(dname, model, mode, alg='avg'):
    # raw and surrogate are ONLY FOR NIPS2022
    configuration_space = CS.ConfigurationSpace()
    fidelity_space = CS.ConfigurationSpace()
    # configuration_space
    if dname == 'twitter':
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('round',
                                         choices=[x for x in range(500)]))
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('sample_rate', choices=[0.01]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter(
                'lr', choices=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter('wd', choices=[0.0, 0.001, 0.01,
                                                        0.1]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter('dropout', choices=[0.0]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter('step', choices=[1, 2, 3, 4]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter('batch', choices=[64]))

    elif dname in ['cora', 'citeseer', 'pubmed']:
        # GNN tabular, raw and surrogate
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('round',
                                         choices=[x for x in range(500)]))
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('sample_rate',
                                         choices=[0.2, 0.4, 0.6, 0.8, 1.0]))
        if mode == 'tabular':
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('lr',
                                             choices=[
                                                 0.01, 0.01668, 0.02783,
                                                 0.04642, 0.07743, 0.12915,
                                                 0.21544, 0.35938, 0.59948, 1.0
                                             ]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('dropout', choices=[0.0, 0.5]))
            if alg == 'avg':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'step', choices=[1, 2, 3, 4, 5, 6, 7, 8]))
            else:
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('lrserver',
                                                 choices=[0.1, 0.5, 1.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('momentumsserver',
                                                 choices=[0.0, 0.9]))
        elif mode in ['surrogate', 'raw']:
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('lr',
                                              lower=1e-2,
                                              upper=1.0,
                                              log=True))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('dropout', lower=.0, upper=.5))
            if alg == 'avg':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'step', choices=[1, 2, 3, 4, 5, 6, 7, 8]))
            else:
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1]))
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('lrserver',
                                                  lower=1e-1,
                                                  upper=1.0,
                                                  log=True))
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('momentumsserver',
                                                  lower=0.0,
                                                  upper=1.0))

    elif dname in [
            '10101@openml', '53@openml', '146818@openml', '146821@openml',
            '9952@openml', '146822@openml', '31@openml', '3917@openml'
    ]:
        # Openml tabular, raw and surrogate
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('round',
                                         choices=[x for x in range(250)]))
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('sample_rate',
                                         choices=[0.2, 0.4, 0.6, 0.8, 1.0]))
        if model == 'lr':
            if mode == 'tabular':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'lr', choices=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                       1.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'wd', choices=[0.0, 0.001, 0.01, 0.1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'batch', choices=[8, 16, 32, 64, 128, 256]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('dropout', choices=[0.0]))
                if alg == 'avg':
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step',
                                                     choices=[1, 2, 3, 4]))
                else:
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step', choices=[1]))
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('lrserver',
                                                     choices=[0.1, 0.5, 1.0]))
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('momentumsserver',
                                                     choices=[0.0, 0.9]))
            elif mode in ['surrogate', 'raw']:
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('lr',
                                                  lower=1e-5,
                                                  upper=1.0,
                                                  log=True))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'wd', choices=[0.0, 0.001, 0.01, 0.1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('dropout', choices=[0.0]))
                if alg == 'avg':
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter(
                            'step', choices=[1, 2, 3, 4, 5, 6, 7, 8]))
                else:
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step', choices=[1]))
                    configuration_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter('lrserver',
                                                      lower=1e-1,
                                                      upper=1.0,
                                                      log=True))
                    configuration_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter('momentumsserver',
                                                      lower=0.0,
                                                      upper=1.0))
        elif model == 'mlp':
            if mode == 'tabular':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'lr', choices=[0.00001, 0.0001, 0.001, 0.01, 0.1,
                                       1.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'wd', choices=[0.0, 0.001, 0.01, 0.1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('batch',
                                                 choices=[32, 64, 128, 256]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('dropout', choices=[0.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('layer', choices=[2, 3, 4]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('hidden',
                                                 choices=[16, 64, 256]))
                if alg == 'avg':
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step',
                                                     choices=[1, 2, 3, 4]))
                else:
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step', choices=[1]))
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('lrserver',
                                                     choices=[0.1, 0.5, 1.0]))
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('momentumsserver',
                                                     choices=[0.0, 0.9]))
            elif mode in ['surrogate', 'raw']:
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('lr',
                                                  lower=1e-5,
                                                  upper=1.0,
                                                  log=True))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'wd', choices=[0.0, 0.001, 0.01, 0.1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('dropout', choices=[0.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('layer', choices=[2, 3, 4]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('hidden',
                                                 choices=[16, 64, 256]))
                if alg == 'avg':
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step',
                                                     choices=[1, 2, 3, 4]))
                else:
                    configuration_space.add_hyperparameter(
                        CS.CategoricalHyperparameter('step', choices=[1]))
                    configuration_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter('lrserver',
                                                      lower=1e-1,
                                                      upper=1.0,
                                                      log=True))
                    configuration_space.add_hyperparameter(
                        CS.UniformFloatHyperparameter('momentumsserver',
                                                      lower=0.0,
                                                      upper=1.0))
    elif dname in ['femnist', 'cifar10']:
        # CNN tabular and surrogate
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('round',
                                         choices=[x for x in range(250)]))
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('sample_rate',
                                         choices=[0.2, 0.4, 0.6, 0.8, 1.0]))
        if mode == 'tabular':
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('lr',
                                             choices=[
                                                 0.01, 0.01668, 0.02783,
                                                 0.04642, 0.07743, 0.12915,
                                                 0.21544, 0.35938, 0.59948, 1.0
                                             ]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('dropout', choices=[0.0, 0.5]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('batch', choices=[16, 32, 64]))
            if alg == 'avg':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1, 2, 3, 4]))
        elif mode in ['surrogate', 'raw']:
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('lr',
                                              lower=1e-2,
                                              upper=1.0,
                                              log=True))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('dropout', lower=.0, upper=.5))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('batch', choices=[16, 32, 64]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('step', choices=[1, 2, 3, 4]))
    elif dname in ['sst2@huggingface_datasets', 'cola@huggingface_datasets']:
        # Transformer tabular and surrogate
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('round',
                                         choices=[x for x in range(40)]))
        fidelity_space.add_hyperparameter(
            CS.CategoricalHyperparameter('sample_rate',
                                         choices=[0.2, 0.4, 0.6, 0.8, 1.0]))
        configuration_space.add_hyperparameter(
            CS.CategoricalHyperparameter('batch', choices=[8, 16, 32, 64,
                                                           128]))
        if mode == 'tabular':
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('lr',
                                             choices=[
                                                 0.01, 0.01668, 0.02783,
                                                 0.04642, 0.07743, 0.12915,
                                                 0.21544, 0.35938, 0.59948, 1.0
                                             ]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('dropout', choices=[0.0, 0.5]))
            if alg == 'avg':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1, 2, 3, 4]))
            else:
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('lrserver',
                                                 choices=[0.1, 0.5, 1.0]))
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('momentumsserver',
                                                 choices=[0.0, 0.9]))
        elif mode in ['surrogate', 'raw']:
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('lr',
                                              lower=1e-2,
                                              upper=1.0,
                                              log=True))
            configuration_space.add_hyperparameter(
                CS.CategoricalHyperparameter('wd',
                                             choices=[0.0, 0.001, 0.01, 0.1]))
            configuration_space.add_hyperparameter(
                CS.UniformFloatHyperparameter('dropout', lower=.0, upper=.5))
            if alg == 'avg':
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter(
                        'step', choices=[1, 2, 3, 4, 5, 6, 7, 8]))
            else:
                configuration_space.add_hyperparameter(
                    CS.CategoricalHyperparameter('step', choices=[1]))
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('lrserver',
                                                  lower=1e-1,
                                                  upper=1.0,
                                                  log=True))
                configuration_space.add_hyperparameter(
                    CS.UniformFloatHyperparameter('momentumsserver',
                                                  lower=0.0,
                                                  upper=1.0))
    return configuration_space, fidelity_space


def initial_cfg(cfg):
    # ---------------------------------------------------------------------- #
    # benchmark related options
    # ---------------------------------------------------------------------- #
    cfg.benchmark = CN()
    cfg.benchmark.cls = [{
        'raw': RawBenchmark,
        'tabular': TabularBenchmark,
        'surrogate': SurrogateBenchmark
    }]

    # ********************************************************************** #
    cfg.benchmark.type = 'raw'
    cfg.benchmark.model = 'gcn'
    cfg.benchmark.data = 'cora'
    cfg.benchmark.device = 0
    cfg.benchmark.sample_client = 1.0  # only for optimizer
    cfg.benchmark.algo = 'avg'  # ['avg', 'opt']
    cfg.benchmark.out_dir = 'exp_results'
    # ********************************************************************** #

    # ---------------------------------------------------------------------- #
    # cost related options
    # ---------------------------------------------------------------------- #
    cfg.cost = CN()
    cfg.cost.type = 'estimated'  # in ['raw', 'estimated']
    cfg.cost.c = 1  # lambda for exponential distribution, time consumed in
    # client
    cfg.cost.time_server = 0  # time consumed in server, `0` for real time
    cfg.cost.lag_const = 65535  # Max port number

    # bandwidth for estimated cost
    cfg.cost.bandwidth = CN()
    cfg.cost.bandwidth.client_up = 0.25 * 1024 * 1024 * 8 / 32
    cfg.cost.bandwidth.client_down = 0.75 * 1024 * 1024 * 8 / 32
    cfg.cost.bandwidth.server_up = 0.25 * 1024 * 1024 * 8 / 32
    cfg.cost.bandwidth.server_down = 0.75 * 1024 * 1024 * 8 / 32

    # ---------------------------------------------------------------------- #
    # optimizer related options
    # ---------------------------------------------------------------------- #
    cfg.optimizer = CN()
    cfg.optimizer.type = 'de'
    cfg.optimizer.min_budget = 1
    cfg.optimizer.max_budget = 243
    cfg.optimizer.n_iterations = 100000000  # No limits
    cfg.optimizer.seed = 1
    cfg.optimizer.limit_time = 86400  # one day

    # ---------------------------------------------------------------------- #
    # hpbandster related options (rs, bo_kde, hb, bohb)
    # ---------------------------------------------------------------------- #
    cfg.optimizer.hpbandster = CN()
    cfg.optimizer.hpbandster.eta = 3
    cfg.optimizer.hpbandster.max_stages = 5

    # ---------------------------------------------------------------------- #
    # smac related options (bo_gp, bo_rf)
    # ---------------------------------------------------------------------- #
    cfg.optimizer.smac = CN()

    # ---------------------------------------------------------------------- #
    # dehb related options (dehb, de)
    # ---------------------------------------------------------------------- #
    cfg.optimizer.dehb = CN()
    cfg.optimizer.dehb.strategy = 'rand1_bin'
    cfg.optimizer.dehb.mutation_factor = 0.5
    cfg.optimizer.dehb.crossover_prob = 0.5

    # dehb.dehb
    cfg.optimizer.dehb.dehb = CN()
    cfg.optimizer.dehb.dehb.gens = 1
    cfg.optimizer.dehb.dehb.eta = 3
    cfg.optimizer.dehb.dehb.async_strategy = 'immediate'

    # dehb.de
    cfg.optimizer.dehb.de = CN()
    cfg.optimizer.dehb.de.pop_size = 20

    # ---------------------------------------------------------------------- #
    # optuna related options (tpe_md, tpe_hb)
    # ---------------------------------------------------------------------- #
    cfg.optimizer.optuna = CN()
    cfg.optimizer.optuna.reduction_factor = 3


def add_configs(cfg):
    # ---------------------------------------------------------------------- #
    # HPO search space related options, which is fixed when mode is `raw`
    # ---------------------------------------------------------------------- #
    configuration_space, fidelity_space = get_cs(cfg.benchmark.data,
                                                 cfg.benchmark.model,
                                                 cfg.benchmark.type,
                                                 cfg.benchmark.algo)

    cfg.benchmark.configuration_space = [configuration_space
                                         ]  # avoid invalid type
    cfg.benchmark.fidelity_space = [fidelity_space]  # avoid invalid type


initial_cfg(fhb_cfg)

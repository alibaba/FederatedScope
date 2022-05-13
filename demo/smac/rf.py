"""
This script is provided to demonstrate the usage of SMAC's Black-box optimization with Random Forest model, where we have assumed the availability of related packages.
More details about SMAC can be found at https://github.com/automl/SMAC3
"""
import numpy as np

from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, UniformFloatHyperparameter, CategoricalHyperparameter
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario


def eval_fl_algo(x):
    from federatedscope.core.cmd_args import parse_args
    from federatedscope.core.auxiliaries.data_builder import get_data
    from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
    from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.fed_runner import FedRunner

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(
        "federatedscope/example_configs/single_process.yaml")
    # specify the configuration of interest
    init_cfg.merge_from_list([
        "optimizer.lr",
        float(x['lr']), "optimizer.weight_decay",
        float(x['wd']), "model.dropout",
        float(x["dropout"])
    ])

    update_logger(init_cfg, True)
    setup_seed(init_cfg.seed)

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone())
    results = runner.run()

    # so that we could modify cfg in the next trial
    init_cfg.defrost()

    return results['client_summarized_weighted_avg']['test_avg_loss']


def main():
    # Define your hyperparameters
    configspace = ConfigurationSpace()
    #configspace.add_hyperparameter(UniformIntegerHyperparameter("depth", 2, 100))
    configspace.add_hyperparameter(
        UniformFloatHyperparameter("lr", lower=1e-4, upper=1.0, log=True))
    configspace.add_hyperparameter(
        UniformFloatHyperparameter("dropout", lower=.0, upper=.5))
    configspace.add_hyperparameter(
        CategoricalHyperparameter("wd", choices=[0.0, 0.5]))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 8,  # Max number of function evaluations (the more the better)
        "cs": configspace,
        'output_dir': "smac_rf",
    })

    # a summary of SMAC's facades: https://automl.github.io/SMAC3/master/pages/details/facades.html?highlight=random%20forest#facades
    smac = SMAC4HPO(scenario=scenario, tae_runner=eval_fl_algo)
    best_found_config = smac.optimize()
    print(best_found_config)
    #run_history = smac.get_runhistory()


if __name__ == "__main__":
    main()

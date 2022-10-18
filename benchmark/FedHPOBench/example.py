from fedhpobench.config import fhb_cfg
from fedhpobench.benchmarks import TabularBenchmark

benchmark = TabularBenchmark('cnn', 'femnist', 'avg')

# get hyperparameters space
config_space = benchmark.get_configuration_space(CS=True)

# get fidelity space
fidelity_space = benchmark.get_fidelity_space(CS=True)

# get results
res = benchmark(config_space.sample_configuration(),
                fidelity_space.sample_configuration(),
                fhb_cfg=fhb_cfg,
                seed=12345)
print(res)

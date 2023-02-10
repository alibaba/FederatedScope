set -e

alpha=$1

bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 0 1e-4 0.01 1e-4 0.75 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 1 1e-4 0.01 1e-4 1.0 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 2 1e-4 0.1 1e-4 0.75 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 3 1e-4 0.1 1e-4 1.0 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 4 0.03 0.01 1e-4 0.75 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 5 0.03 0.01 1e-4 1.0 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 6 0.03 0.1 1e-4 0.75 False >/dev/null 2>/dev/null &
bash scripts/wide_valley_exp_scripts/run_fedentsgd_on_cifar10.sh $alpha 7 0.03 0.1 1e-4 1.0 False >/dev/null 2>/dev/null &

set -e

bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 0 1e-4 1e-4 0.1 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 1 1e-4 1e-4 1.0 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 2 1e-4 1e-3 0.1 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 3 1e-4 1e-3 1.0 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 4 1e-3 1e-4 0.1 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 5 1e-3 1e-4 1.0 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 6 1e-3 1e-3 0.1 >/dev/null 2>/dev/null &
bash scripts/fedsam_exp_scripts/run_fedentsgd_on_cifar10.sh 0.05 7 1e-3 1e-3 1.0 >/dev/null 2>/dev/null &

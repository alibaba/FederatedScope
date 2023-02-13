set -e

# Seed 12345
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.224973 train.optimizer.weight_decay 0.001659 device 0 seed 12345 >rs_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 1 train.optimizer.lr 0.7170358969751284 train.optimizer.weight_decay 0.0 device 1 seed 12345 >rs_wrap_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 4 train.optimizer.lr 0.999311 train.optimizer.weight_decay 8.674074e-07 device 2 seed 12345 >bo_gp_12345.log &
# >rs_wrap_12345.log >bo_gp_wrap_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 7 train.optimizer.lr 0.46701 train.optimizer.weight_decay 0.000853 device 0 seed 12345 >bo_kde_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 7 train.optimizer.lr 0.46852259676975294 train.optimizer.weight_decay 0.263664919539593 device 1 seed 12345 >bo_kde_wrap_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 6 train.optimizer.lr 0.984828 train.optimizer.weight_decay 0.00233 device 2 seed 12345 >bo_rf_12345.log &
# >rs_wrap_12345.log >bo_rf_wrap_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.224973 train.optimizer.weight_decay 0.001659 device 0 seed 12345 >hb_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 4 train.optimizer.lr 0.453625286588245 train.optimizer.weight_decay 0.06289557213350952 device 1 seed 12345 >hb_wrap_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.224973 train.optimizer.weight_decay 0.001659 device 2 seed 12345 >bohb_12345.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 6 train.optimizer.lr 0.14495809658233344 train.optimizer.weight_decay 0.04477752881699114 device 3 seed 12345 >bohb_wrap_12345.log &

# Seed 12346
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 8 train.optimizer.lr 0.75764 train.optimizer.weight_decay 0.007947 device 0 seed 12346 >rs_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 7 train.optimizer.lr 0.11480094171726124 train.optimizer.weight_decay 0.15500565663984123 device 1 seed 12346 >rs_wrap_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 3 train.optimizer.lr 0.999636 train.optimizer.weight_decay 9.7e-05 device 2 seed 12346 >bo_gp_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 2 train.optimizer.lr 0.6231045010965764 train.optimizer.weight_decay 0.020651830583413487 device 3 seed 12346 >bo_gp_wrap_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.094777 train.optimizer.weight_decay 0.019946 device 0 seed 12346 >bo_kde_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.09470439245585746 train.optimizer.weight_decay 0.1134997725167636 device 1 seed 12346 >bo_kde_wrap_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 3 train.optimizer.lr 0.991872 train.optimizer.weight_decay 0.000695 device 2 seed 12346 >bo_rf_12346.log &
# >bo_gp_wrap_12346.log >bo_rf_wrap_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 8 train.optimizer.lr 0.75764 train.optimizer.weight_decay 0.007947 device 0 seed 12346 >hb_12346.log &
# >bo_kde_wrap_12346.log >hb_wrap_12346.log &
# >hb_12346.log >bohb_12346.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 7 train.optimizer.lr 0.19783377983946915 train.optimizer.weight_decay 0.3087280858375391 device 3 seed 12346 >bohb_wrap_12346.log &

# Seed 12347
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 4 train.optimizer.lr 0.110196 train.optimizer.weight_decay 0.028109 device 0 seed 12347 >rs_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 3 train.optimizer.lr 0.043324742759307006 train.optimizer.weight_decay 0.0 device 1 seed 12347 >rs_wrap_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 8 train.optimizer.lr 0.999856 train.optimizer.weight_decay 1.2e-05 device 2 seed 12347 >bo_gp_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 2 train.optimizer.lr 0.05512741305783294 train.optimizer.weight_decay 0.32323708666212847 device 3 seed 12347 >bo_gp_wrap_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 2 train.optimizer.lr 0.020666 train.optimizer.weight_decay 0.010875 device 0 seed 12347 >bo_kde_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 1 train.optimizer.lr 0.029033027691434364 train.optimizer.weight_decay 0.7362670096896027 device 1 seed 12347 >bo_kde_wrap_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 8 train.optimizer.lr 0.998016 train.optimizer.weight_decay 0.001249 device 2 seed 12347 >bo_rf_12347.log &
# >bo_gp_wrap_12347.log >bo_rf_wrap_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 4 train.optimizer.lr 0.110196 train.optimizer.weight_decay 0.028109 device 0 seed 12347 >hb_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 5 train.optimizer.lr 0.07982026000525726 train.optimizer.weight_decay 0.07891495031558063 device 1 seed 12347 >hb_wrap_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.5 train.local_update_steps 7 train.optimizer.lr 0.015962 train.optimizer.weight_decay 1.5e-05 device 2 seed 12347 >bohb_12347.log &
nohup python federatedscope/main.py --cfg scripts/hpo_exp_scripts/usability/pubmed/learn_from_scratch/pubmed.yaml model.dropout 0.0 train.local_update_steps 1 train.optimizer.lr 0.04927136943461437 train.optimizer.weight_decay 0.08747080914279645 device 3 seed 12347 >bohb_wrap_12347.log &

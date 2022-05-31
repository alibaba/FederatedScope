# --Seed 1 --
# --1--
bash run_hpo_glue.sh 0 0.2 sst2 1 &
bash run_hpo_glue.sh 1 0.4 sst2 1 &
bash run_hpo_glue.sh 2 0.6 sst2 1 &
bash run_hpo_glue.sh 3 0.8 sst2 1 &

# --2--
bash run_hpo_glue.sh 0 1.0 sst2 1 &
bash run_hpo_glue.sh 1 0.2 cola 1 &
bash run_hpo_glue.sh 2 0.4 cola 1 &
bash run_hpo_glue.sh 3 0.6 cola 1 &

# --3--
bash run_hpo_glue.sh 0 0.8 cola 1 &
bash run_hpo_glue.sh 1 1.0 cola 1 &
bash run_opt_glue.sh 2 0.2 sst2 1 &
bash run_opt_glue.sh 3 0.4 sst2 1 &

# --4--
bash run_opt_glue.sh 0 0.6 sst2 1 &
bash run_opt_glue.sh 1 0.8 sst2 1 &
bash run_opt_glue.sh 2 1.0 sst2 1 &
bash run_opt_glue.sh 3 0.2 cola 1 &

# --5--
bash run_opt_glue.sh 0 0.4 cola 1 &
bash run_opt_glue.sh 1 0.6 cola 1 &
bash run_opt_glue.sh 2 0.8 cola 1 &
bash run_opt_glue.sh 3 1.0 cola 1 &





# --Seed 2 --
# --1--
bash run_hpo_glue.sh 0 0.2 sst2 2 &
bash run_hpo_glue.sh 1 0.4 sst2 2 &
bash run_hpo_glue.sh 2 0.6 sst2 2 &
bash run_hpo_glue.sh 3 0.8 sst2 2 &

# --2--
bash run_hpo_glue.sh 0 1.0 sst2 2 &
bash run_hpo_glue.sh 1 0.2 cola 2 &
bash run_hpo_glue.sh 2 0.4 cola 2 &
bash run_hpo_glue.sh 3 0.6 cola 2 &

# --3--
bash run_hpo_glue.sh 0 0.8 cola 2 &
bash run_hpo_glue.sh 1 1.0 cola 2 &
bash run_opt_glue.sh 2 0.2 sst2 2 &
bash run_opt_glue.sh 3 0.4 sst2 2 &

# --4--
bash run_opt_glue.sh 0 0.6 sst2 2 &
bash run_opt_glue.sh 1 0.8 sst2 2 &
bash run_opt_glue.sh 2 1.0 sst2 2 &
bash run_opt_glue.sh 3 0.2 cola 2 &

# --5--
bash run_opt_glue.sh 0 0.4 cola 2 &
bash run_opt_glue.sh 1 0.6 cola 2 &
bash run_opt_glue.sh 2 0.8 cola 2 &
bash run_opt_glue.sh 3 1.0 cola 2 &



# --Seed 3 --
# --1--
bash run_hpo_glue.sh 0 0.2 sst2 3 &
bash run_hpo_glue.sh 1 0.4 sst2 3 &
bash run_hpo_glue.sh 2 0.6 sst2 3 &
bash run_hpo_glue.sh 3 0.8 sst2 3 &

# --2--
bash run_hpo_glue.sh 0 1.0 sst2 3 &
bash run_hpo_glue.sh 1 0.2 cola 3 &
bash run_hpo_glue.sh 2 0.4 cola 3 &
bash run_hpo_glue.sh 3 0.6 cola 3 &

# --3--
bash run_hpo_glue.sh 0 0.8 cola 3 &
bash run_hpo_glue.sh 1 1.0 cola 3 &
bash run_opt_glue.sh 2 0.2 sst2 3 &
bash run_opt_glue.sh 3 0.4 sst2 3 &

# --4--
bash run_opt_glue.sh 0 0.6 sst2 3 &
bash run_opt_glue.sh 1 0.8 sst2 3 &
bash run_opt_glue.sh 2 1.0 sst2 3 &
bash run_opt_glue.sh 3 0.2 cola 3 &

# --5--
bash run_opt_glue.sh 0 0.4 cola 3 &
bash run_opt_glue.sh 1 0.6 cola 3 &
bash run_opt_glue.sh 2 0.8 cola 3 &
bash run_opt_glue.sh 3 1.0 cola 3 &

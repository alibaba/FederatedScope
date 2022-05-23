bash run_flitplus.sh 1 bbbp flitplustrainer 0.01 0.1 &

bash run_flitplus.sh 2 bbbp flittrainer 0.1 &

bash run_flitplus.sh 3 bbbp fedfocaltrainer 0.1 &

bash run_flitplus.sh 4 bbbp fedvattrainer 0.1 0.1 &

bash run_flitplus.sh 5 bbbp graphminibatch_trainer 0.1 &

bash run_fedprox.sh 6 bbbp graphminibatch_trainer 0.1 0.1 &
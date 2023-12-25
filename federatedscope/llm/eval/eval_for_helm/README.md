# Helm + FS

## Docker

* Build images:
  * Build from Dockerfile: `docker build -f federatedscope-torch2.0-helm.Dockerfile -t fsteam/federatedscope:fs_helm .`
  * Pull from docker hub: `docker pull fsteam/federatedscope:fs_helm`

* Download Helm evaluation dataset

  * `wget https://federatedscope.oss-cn-beijing.aliyuncs.com/helm_data.zip -O ${PATH_TO_HELM_DATA}/helm_data.zip`
  * `unzip ${PATH_TO_HELM_DATA}/helm_data.zip`

* Prepare FS and related `ckpt` and `yaml`

  * `${PATH_TO_FS}`

* Launch and mapping dataset and FS

  ```bash
  docker run -p ${PORT}:${DOCKER_PORT} -u root: --gpus device=all -it --rm \
  -v "${PATH_TO_HELM_DATA}/helm_data/benchmark_output:/root/src/helm/benchmark_output" \
  -v "${PATH_TO_HELM_DATA}/helm_data/nltk_data:/root/nltk_data" \
  -v "${PATH_TO_HELM_DATA}/helm_data/prompt_construction_settings.json:/tmp/prompt_construction_settings.json" \
  -v "${PATH_TO_FS}:/root/FederatedScope" \
  -v "${PATH_TO_CACHE}:/root/.cache" \
  -w '/root/FederatedScope' \
  --name "helm_fs" fsteam/federatedscope:fs_helm /bin/bash
  ```

  Example for a root user:

  ```bash
  docker run -p 8000:8000 -u root: --gpus device=all -it --rm \
  -v "/root/helm_fs/helm_data/benchmark_output:/root/src/helm/benchmark_output" \
  -v "/root/helm_fs/helm_data/nltk_data:/root/nltk_data" \
  -v "/root/helm_fs/helm_data/prompt_construction_settings.json:/tmp/prompt_construction_settings.json" \
  -v "/root/helm_fs/FederatedScope:/root/FederatedScope" \
  -v "/root/.cache:/root/.cache" \
  -w '/root/FederatedScope' \
  --name "helm_fs" fsteam/federatedscope:fs_helm /bin/bash
  ```

* Install FS in container

  * `pip install -e .[llm]`

* Move to helm

  * `cd /root/src/crfm-helm`

* Start to evaluate

  * `helm-run --conf-paths federatedscope/llm/eval/eval_for_helm/run_specs.conf --enable-local-huggingface-model decapoda-research/llama-7b-hf --suite ${SUITE_NAME} -m 100 --local -n 1 --skip-completed-runs --local-path xxx` 
    * The above code will evaluate the model `decapoda-research/llama-7b-hf` and save the results in `/benchmark_output/runs/${SUITE_NAME}`. 
    * `-m 100` means that there will be 100 items in each task.
    * `--skip-completed-runs` means that when restarted, it will skip the completed test sets. It is recommended to add this if you no dot want to waste your time for the completed tasks.
    * `--local-path xxx` means the directory to put cache files, default value is `prod_env`. It will always use it when you run a new task. It is recommended that before running a new task, delete it or assign a new name to it.
    * If you want to test your own trained `ckpt` for `decapoda-research/llama-7b-hf`, please add parameters `--yaml /path/to/xxx.yaml`. If you want to modify the configurations in `yaml`, just add parameters similar to the behaviors in FS. For example, add `federate.save_to xxxx.ckpt` to change the ckpt. 
* Launch webserver to view results
  * `bash evaluaton/setup_server.sh -n ${SUITE_NAME} -p ${PORT}`

    Run the above code and view the results on port `${PORT}`.
  * Remark: Actually, it will always show the results of the last task. If you want to see the results of another task, say, the suite name is `result_of_exp1`, add `?suite=result_of_exp1` after the port address.

## Conda

* Create new env `helm_fs` in conda
  * `conda create -n helm_fs python=3.9`
* Create dir
  * `mkdir helm_fs` 
  * `cd helm_fs`
* Install helm from our branch
  * `pip install -e git+https://github.com/qbc2016/helm.git@helm_for_fs#egg=crfm-helm`
* Install FS-LLM (**errors can be igored**)
  * `git clone -b llm https://github.com/alibaba/FederatedScope.git`
  * `cd FederatedScope`
  * `pip install -e .[llm]`
* Download and unzip Helm evaluation dataset
  * `wget https://federatedscope.oss-cn-beijing.aliyuncs.com/helm_data.zip -O ${PATH_TO_HELM_DATA}/helm_data.zip`
  * `unzip ${PATH_TO_HELM_DATA}/helm_data.zip`
* Move files
  * `benchmark_output` -> `~/helm_fs/src/crfm-helm/benchmark_output`
  * `nltk_data` -> `~/nltk_data`
  * `prompt_construction_settings.json` - > `/tmp/prompt_construction_settings.json`
* Move ckpt and yaml
* Start to evaluate
  * `helm-run --conf-paths federatedscope/llm/eval/eval_for_helm/run_specs.conf --enable-local-huggingface-model decapoda-research/llama-7b-hf --suite ${SUITE_NAME} -m 100 --local -n 1 --skip-completed-runs --local-path xxx`
* Launch webserver to view results
  * In `~/helm_fs/src/crfm-helm/evaluation/setup_server.sh`, set 
    * `SUITE_NAME=${SUITE_NAME}`
    * `PATH_HELM=~/helm_fs/src/crfm-helm`
    * `PATH_WORKDIR=~/helm_fs/src/crfm-helm`
    * `root/miniconda3/bin/python -> ${which python}`
  * `bash evaluation/setup_server.sh -n ${SUITE_NAME} -p ${PORT}`
    * Remark: Actually, it will show the result of the last task. If you want to see the result of another task, say, the suite name is result_of_exp1, add `?suite=result_of_exp1`after the port address.

Remark: For the second run of `decapoda-research/llama-7b-hf`, if not work, in ~/helm_fs/src/crfm-helm/data/decapoda-research--llama-7b-hf/snapshots/xxxx/tokenizer_config.json, change

"tokenizer_class": "LLaMATokenizer" -> "tokenizer_class": "LlamaTokenizer"

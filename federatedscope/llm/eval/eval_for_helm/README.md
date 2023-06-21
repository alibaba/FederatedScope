# Helm + FS

## Docker

* Build images:
  * Build from Dockerfile: `docker build -f federatedscope-torch2.0-helm.Dockerfile -t alibaba/federatedscope:helm .`
  * Pull from docker hub: `TBD`

* Download Helm evaluation dataset

  * `wget https://${NOT_AVAILABLE_NOW}/helm_data.zip -O ${PATH_TO_HELM_DATA}/helm_data.zip`
  * `unzip ${PATH_TO_HELM_DATA}/helm_data.zip`

* Prepare FS and related `ckpt` and `yaml`

  * `${PATH_TO_FS}`

* Launch and mapping dataset and FS

  ```bash
  docker run -u root: --gpus device=all -it --rm \
  -v "${PATH_TO_HELM_DATA}/helm_data/benchmark_output:/root/src/helm/benchmark_output" \
  -v "${PATH_TO_HELM_DATA}/helm_data/nltk_data:/root/nltk_data" \
  -v "${PATH_TO_HELM_DATA}/helm_data/prompt_construction_settings.json:/tmp/prompt_construction_settings.json" \
  -v "${PATH_TO_FS}:/root/FederatedScope" \
  -v "${PATH_TO_CACHE}:/root/.cache" \
  -w '/root/FederatedScope' \
  --name "helm_fs" alibaba/federatedscope:helm /bin/bash
  ```

  Example for a root user:

  ```bash
  docker run -u root: --gpus device=all -it --rm \
  -v "/root/helm_fs/helm_data/benchmark_output:/root/src/helm/benchmark_output" \
  -v "/root/helm_fs/helm_data/nltk_data:/root/nltk_data" \
  -v "/root/helm_fs/helm_data/prompt_construction_settings.json:/tmp/prompt_construction_settings.json" \
  -v "/root/helm_fs/FederatedScope:/root/FederatedScope" \
  -v "/root/.cache:/root/.cache" \
  -w '/root/FederatedScope' \
  --name "helm_fs" alibaba/federatedscope:helm /bin/bash
  ```

* Install FS in container

  * `pip install -e .[llm]`

* Move to helm

  * `cd /root/helm`

* Start to evaluate

  * `helm-run --conf-paths federatedscope/llm/eval/eval_for_helm/run_specs.conf --enable-local-huggingface-model decapoda-research/llama-7b-hf --suite test -m 100 --local -n 1 --yaml federatedscope/llm/baseline/llama.yaml`

## Conda

* Create new env `helm_fs` in conda
  * `conda create -n helm_fs python=3.9`
* Create dir
  * `mkdir helm_fs` 
  * `cd helm_fs`
* Install helm from our branch
  * `pip install -e git+https://github.com/qbc2016/helm.git@helm_for_fs#egg=crfm-helm`
* Install FS-LLM (**errors can be igored**)
  * `git clone -b dev/llm https://github.com/alibaba/FederatedScope.git`
  * `cd FederatedScope`
  * `pip install -e .[llm]`
* Unzip `helm_data.zip` and move data
  * `benchmark_output` -> `~/helm_fs/src/crfm-helm/benchmark_output`
  * `nltk_data` -> `~/nltk_data`
  * `prompt_construction_settings.json` - > `/tmp/prompt_construction_settings.json`
* Move ckpt and yaml
* Start to evaluate
  * `helm-run --conf-paths federatedscope/llm/eval/eval_for_helm/run_specs.conf --enable-local-huggingface-model decapoda-research/llama-7b-hf --suite test -m 100 --local -n 1 --yaml federatedscope/llm/baseline/llama.yaml --ckpt_dir xxxx --skip-completed-runs --local-path xxx`
    * If the program terminated due to network issues, --skip-completed-runs means that when restart, it will skip the completed test sets. It is recommended to add this all the time.
    * --local-path xxx means the directory to put cache files, default value is prod_env. It will always use it when you run a new task. It is recommended that before running a new task, delete it or assign a new name to it.
* Launch webserver to view results
  * In ~/helm_fs/src/crfm-helm/evaluation/setup_server.sh, set
    * `SUITE_NAME=${suite}`
    * `PATH_HELM=~/helm_fs/src/crfm-helm`
    * `PATH_HELM=~/helm_fs/src/crfm-helm`
    * `root/miniconda3/bin/python -> ${which python}`
  * `bash evaluation/setup_server.sh`
    * Remark: Actually, it will show the result of the last task. If you want to see the result of another task, say, the suite name is result_of_exp1, add `?suite=result_of_exp1`after the port address.

Remark: For the second run of decapoda-research/llama-7b-hf, it not work, in ~/helm_fs/src/crfm-helm/data/decapoda-research--llama-7b-hf/snapshots/xxxx/tokenizer_config.json, change

"tokenizer_class": "LLaMATokenizer" -> "tokenizer_class": "LlamaTokenizer"
# Helm + FS

* Build images:
  * Build from Dockerfile: `docker build -f federatedscope-torch2.0-helm.Dockerfile -t alibaba/federatedscope:helm .`
  * Pull from docker hub: `TBD`

* Download Helm evaluation dataset

  * `wget ?????/helm_data.zip -O ${PATH_TO_HELM_DATA}/helm_data.zip`
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

* Start to evaluate
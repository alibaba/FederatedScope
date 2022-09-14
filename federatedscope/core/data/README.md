# DataZoo

FederatedScope provides a rich collection of federated datasets for researchers, including images, texts, graphs, recommendation systems, and speeches, as well as utility classes `BaseDataTranslator` for building your own FS datasets.

## Built-in FS data

All datasets can be accessed from [`federatedscope.core.auxiliaries.data_builder.get_data`](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py), which are built to [`federatedscope.core.data.StandaloneDataDict`](https://github.com/alibaba/FederatedScope/tree/master/federatedscope/core/data/base_data.py) (for more details, see [[DataZoo advanced]](#advanced)). By setting `cfg.data.type = DATASET_NAME`, FS would download and pre-process a specific dataset to be passed to `FedRunner`. For example:

```python
# Source: federatedscope/main.py

data, cfg = get_data(cfg)
runner = FedRunner(data=data,
                   server_class=get_server_cls(cfg),
                   client_class=get_client_cls(cfg),
                   config=cfg.clone())
```

We provide a **look-up table** for you to get started with our DataZoo:

| `cfg.data.type`                                              | Domain              |
| ------------------------------------------------------------ | ------------------- |
| [FEMNIST](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/cv/dataset/leaf_cv.py) | CV                  |
| [Celeba](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/cv/dataset/leaf_cv.py) | CV                  |
| [{DNAME}@torchvision](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py) | CV                  |
| [Shakespeare](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/nlp/dataset/leaf_nlp.py) | NLP                 |
| [SubReddit](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/nlp/dataset/leaf_nlp.py) | NLP                 |
| [Twitter (Sentiment140)](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/nlp/dataset/leaf_twitter.py) | NLP                 |
| [{DNAME}@torchtext](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py) | NLP                 |
| [{DNAME}@huggingface_datasets](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py) | NLP                 |
| [Cora](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_node.py) | Graph (node-level)  |
| [CiteSeer](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_node.py) | Graph (node-level)  |
| [PubMed](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_node.py) | Graph (node-level)  |
| [DBLP_conf](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/dblp_new.py) | Graph (node-level)  |
| [DBLP_org](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/dblp_new.py) | Graph (node-level)  |
| [csbm](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/cSBM_dataset.py) | Graph (node-level)  |
| [Epinions](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/recsys.py) | Graph (link-level)  |
| [Ciao](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/recsys.py) | Graph (link-level)  |
| [FB15k](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_link.py) | Graph (link-level)  |
| [FB15k-237](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_link.py) | Graph (link-level)  |
| [WN18](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_link.py) | Graph (link-level)  |
| [MUTAG](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [BZR](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [COX2](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [DHFR](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [PTC_MR](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [AIDS](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [NCI1](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [ENZYMES](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [DD](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [PROTEINS](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [COLLAB](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [IMDB-BINARY](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [IMDB-MULTI](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [REDDIT-BINARY](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [HIV](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [ESOL](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [FREESOLV](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [LIPO](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [PCBA](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [MUV](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [BACE](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [BBBP](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [TOX21](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [TOXCAST](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [SIDER](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [CLINTOX](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [graph_multi_domain_mol](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [graph_multi_domain_small](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [graph_multi_domain_biochem](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py) | Graph (graph-level) |
| [cikmcup](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataset/cikm_cup.py) | Graph (graph-level) |
| [toy](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py) | Tabular             |
| [synthetic](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/nlp/dataset/leaf_synthetic.py) | Tabular             |
| [quadratic](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/tabular/dataloader/quadratic.py) | Tabular             |
| [{DNAME}openml](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/data_builder.py) | Tabular             |
| [vertical_fl_data](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/vertical_fl/dataloader/dataloader.py) | Tabular(vertical)   |
| [VFLMovieLens1M](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/movielens.py) | Recommendation      |
| [VFLMovieLens10M](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/movielens.py) | Recommendation      |
| [HFLMovieLens1M](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/movielens.py) | Recommendation      |
| [HFLMovieLens10M](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/movielens.py) | Recommendation      |
| [VFLNetflix](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/netflix.py) | Recommendation      |
| [HFLNetflix](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/mf/dataset/netflix.py) | Recommendation      |

## <span id="advanced">DataZoo Advanced</span>

In this section, we will introduce key concepts and tools to help you understand how FS data works and how to use it to build your own data in FS.

Concepts:

* [`federatedscope.core.data.ClientData`](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/data/base_data.py)

  * `ClientData` is a subclass of `dict`. In federated learning, each client (server) owns a `ClientData` for training, validating, or testing. Thus, each `ClientData` has one or more of `train`, `val`, and `test` as keys, and `DataLoader` accordingly. 

  * The `DataLoader` of each key is created by `setup()` method, which specifies the arguments of `DataLoader`, such as `batch_size`, `shuffle` of `cfg`.

    Example:

    ```python
    # Instantiate client_data for each Client
    client_data = ClientData(DataLoader, 
                             cfg, 
                             train=train_data, 
                             val=None, 
                             test=test_data)
    # other_cfg with different batch size
    client_data.setup(other_cfg)
    print(client_data)
    
    >> {'train': DataLoader(train_data), 'test': DataLoader(test_data)}
    ```

* [`federatedscope.core.data.StandaloneDataDict`](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/data/base_data.py)
  * `StandaloneDataDict` is a subclass of `dict`. As the name implies, `StandaloneDataDict` consists of all `ClientData` with client index as key (`0`, `1`, `2`, ...) in standalone mode. The key `0` is the data of the server for global evaluation or other usages. 
  * The method `preprocess()` in `StandaloneDataDict` makes changes to inner `ClientData` when `cfg` changes, such as in global mode, we set `cfg.federate.method == "global"`, and `StandaloneDataDict` will merge all `ClientData` to one client to perform global training.

Tools

* [`federatedscope.core.data.BaseDataTranslator`](https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/data/base_translator.py)

  * `BaseDataTranslator` converts [`torch.utils.data.Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) or `dict` of data split to `StandaloneDataDict` according to `cfg`. After translating, it can be directly passed to `FedRunner` to launch a FL course.

  *  `BaseDataTranslator` will split data to `train`, `val,` and `test` by `cfg.data.splits` (**ML split**). And using `Splitter` to split each data split to each client (**FL split**). In order to use `BaseDataTranslator`, `cfg.data.splitter`, `cfg.federate.client_num,` and other arguments of `Splitter` must be specified.

    Example:

    ```python
    cfg.data.splitter = 'lda'
    cfg.federate.client_num = 5
    cfg.data.splitter_args = [{'alpha': 0.2}]
    
    translator = BaseDataTranslator(global_cfg, DataLoader)
    raw_data = CIFAR10()
    fs_data = translator(raw_data)
    
    runner = FedRunner(data=fs_data,
                       server_class=get_server_cls(cfg),
                       client_class=get_client_cls(cfg),
                       config=cfg.clone())
    ```

* [`federatedscope.core.splitters`](federatedscope.core.splitters)

  * To generate simulated federation datasets, we provide `splitter` who are responsible for dispersing a given standalone dataset into multiple clients, with configurable statistical heterogeneity among them.

  We provide a **look-up table** for you to get started with our `Splitter`:

  | `cfg.data.splitter` | Domain              | Arguments                                        |
  | :------------------ | ------------------- | :----------------------------------------------- |
  | LDA                 | Generic             | `alpha`                                          |
  | Louvain             | Graph (node-level)  | `delta`                                          |
  | Random              | Graph (node-level)  | `sampling_rate`, `overlapping_rate`, `drop_edge` |
  | rel_type            | Graph (link-level)  | `alpha`                                          |
  | Scaffold            | Molecular           | -                                                |
  | Scaffold_lda        | Molecular           | `alpha`                                          |
  | Rand_chunk          | Graph (graph-level) | -                                                |
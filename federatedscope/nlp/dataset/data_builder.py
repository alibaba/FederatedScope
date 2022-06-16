from torch.utils.data import DataLoader
from federatedscope.register import register_data
from federatedscope.nlp.auxiliaries.utils import setup_tokenizer
from federatedscope.nlp.dataset.sts import create_sts_dataset
from federatedscope.nlp.dataset.imdb import create_imdb_dataset
from federatedscope.nlp.dataset.squad import create_squad_dataset


def load_textdt_data(config):
    model_type = config.model.bert_type
    tokenizer = setup_tokenizer(config)

    # sts-b
    root = config.data.dir.sts
    max_seq_len = config.data.max_seq_len.sts
    sts_train, sts_dev = create_sts_dataset(root, 'train', tokenizer, max_seq_len,
                                            model_type=model_type, cache_dir=config.data.cache_dir)
    sts_test = create_sts_dataset(root, 'dev', tokenizer, max_seq_len, model_type=model_type,
                                  cache_dir=config.data.cache_dir)

    # imdb
    root = config.data.dir.imdb
    max_seq_len = config.data.max_seq_len.imdb
    imdb_train, imdb_dev = create_imdb_dataset(root, 'train', tokenizer, max_seq_len,
                                               model_type=model_type, cache_dir=config.data.cache_dir)
    imdb_test = create_imdb_dataset(root, 'test', tokenizer, max_seq_len, model_type=model_type,
                                    cache_dir=config.data.cache_dir)

    # squad
    root = config.data.dir.squad
    max_seq_len = config.data.max_seq_len.squad
    max_query_len = config.data.max_query_len
    trunc_stride = config.data.trunc_stride
    squad_train, squad_dev = create_squad_dataset(root, 'train', tokenizer, max_seq_len, max_query_len,
                                                  trunc_stride, model_type=model_type, cache_dir=config.data.cache_dir)
    squad_test = create_squad_dataset(root, 'dev', tokenizer, max_seq_len, max_query_len,
                                      trunc_stride, model_type=model_type, cache_dir=config.data.cache_dir)

    train_data = [sts_train, imdb_train, squad_train]
    dev_data = [sts_dev, imdb_dev, squad_dev]
    test_data = [sts_test, imdb_test, squad_test]

    data_dict = dict()
    for client_idx in range(1, config.federate.client_num + 1):
        dataloader_dict = {
            'train': {'dataloader': DataLoader(train_data[client_idx - 1][0],
                                               batch_size=config.data.batch_size,
                                               shuffle=config.data.shuffle,
                                               num_workers=config.data.num_workers,
                                               pin_memory=config.use_gpu),
                      'encoded': train_data[client_idx - 1][1],
                      'examples': train_data[client_idx - 1][2]},
            'val': {'dataloader': DataLoader(dev_data[client_idx - 1][0],
                                             batch_size=config.data.batch_size,
                                             shuffle=False,
                                             num_workers=config.data.num_workers,
                                             pin_memory=config.use_gpu),
                    'encoded': dev_data[client_idx - 1][1],
                    'examples': dev_data[client_idx - 1][2]},
            'test': {'dataloader': DataLoader(test_data[client_idx - 1][0],
                                              batch_size=config.data.batch_size,
                                              shuffle=False,
                                              num_workers=config.data.num_workers,
                                              pin_memory=config.use_gpu),
                     'encoded': test_data[client_idx - 1][1],
                     'examples': test_data[client_idx - 1][2]},
        }
        data_dict[client_idx] = dataloader_dict

    return data_dict, config


def call_textdt_data(config):
    if config.data.type == "text-dt":
        data, modified_config = load_textdt_data(config)
        return data, modified_config


register_data('text-dt', call_textdt_data)

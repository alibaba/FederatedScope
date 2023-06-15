import os
import pickle

from federatedscope.register import register_data
from federatedscope.core.data.utils import convert_data_mode
from federatedscope.core.auxiliaries.utils import setup_seed


def load_data_from_file(config, client_cfgs=None):
    from federatedscope.core.data import DummyDataTranslator

    file_path = config.data.file_path

    if not os.path.exists(file_path):
        raise ValueError(f'The file {file_path} does not exist.')

    with open(file_path, 'br') as file:
        data = pickle.load(file)
    # The shape of data is expected to be:
    # (1) the data consist of all participants' data:
    # {
    #   'client_id': {
    #       'train/val/test': {
    #           'x/y': np.ndarray
    #       }
    #   }
    # }
    # (2) isolated data
    # {
    #   'train/val/test': {
    #       'x/y': np.ndarray
    #   }
    # }

    # translator = DummyDataTranslator(config, client_cfgs)
    # data = translator(data)

    # Convert `StandaloneDataDict` to `ClientData` when in distribute mode
    data = convert_data_mode(data, config)

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, config


def call_file_data(config, client_cfgs):
    if config.data.type == "file":
        # All the data (clients and servers) are loaded from one unified files
        data, modified_config = load_data_from_file(config, client_cfgs)
        return data, modified_config


register_data("file", call_file_data)

import numpy as np


def client_selection(num_clients, client_selection_frac, seed=42, other_info=None):
    if seed == 42:
        np.random.seed(other_info)
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))
    else:
        np.random.seed(other_info*(seed+200)*20)
        num_selected = max(int(client_selection_frac * num_clients), 1)
        selected_clients_set = set(np.random.choice(np.arange(num_clients), num_selected, replace=False))

    return selected_clients_set

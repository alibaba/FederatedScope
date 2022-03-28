"""Configuration file for composition of different aggregators, messages, handlers, etc."""


AGGREGATOR_TYPE = {
    "local": "no_communication",
    "fedavg": "clients_avg",  # FedAvg
    "pfedme": "server_clients_interpolation",  # pFedMe,  + server-clients interpolation
    "ditto": "clients_avg",  # Ditto
    "fedsageplus": "clients_avg",
    "gcflplus": "clients_avg",
    "fedopt": "fedopt"
}

CLIENTS_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal_loss_regular",  # pFedMe, + regularization-based local loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
}

SERVER_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal",  # pFedMe, + regularization-based local loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
}

SERVER_TYPE = {
    "local": "normal",
    "fedavg": "normal",  # FedAvg
    "pfedme": "normal",  # pFedMe, + regularization-based local loss
    "ditto": "normal",  # Ditto, + local training for distinct personalized models
    "fedsageplus": "fedsageplus",  # FedSage+ for graph data
    "gcflplus": "gcflplus",  # GCFL+ for graph data
}

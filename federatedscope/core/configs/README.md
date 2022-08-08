## Configurations
We summarize all the customizable configurations here.

### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| data.root | (string) 'data' | <font size=1> The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. </font> | - |


### FL Setting
The configurations related to FL settings are defined in `cfg_fl_setting.py`.
#### `federate`: basic fl setting
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `federate.client_num` | (int) 0 | The number of clients that involves in the FL courses. | It can set to 0 to automatically specify by the partition of dataset. |
| `federate.sample_client_num` | (int) -1 | The number of sampled clients in each training round. | - |
| `federate.sample_client_rate` | (float) -1.0 | The ratio of sampled clients in each training round. | - |
| `federate.unseen_clients_rate` | (float) 0.0 | The ratio of clients served as unseen clients, which would not be used for training and only for evaluation. | - |
| `federate.total_round_num` | (int) 50 | The maximum training round number of the FL course. | - |
| `federate.mode` | (string) 'standalone' </br> Choices: {'standalone', 'distributed'} | The running mode of the FL course. | - |
| `federate.share_local_model` | (bool) False | If `True`, only one model object is created in the FL course and shared among clients for efficient simulation. | - | 
| `federate.data_weighted_aggr` | (bool) False | If `True`, the weight of aggregator is the number of training samples in dataset. | - |
| `federate.online_aggr` | (bool) False | If `True`, an online aggregation mechanism would be applied for efficient simulation. | - | 
| `federate.make_global_eval` | (bool) False | If `True`, the evaluation would be performed on the server's test data, otherwise each client would perform evaluation on local test set and the results would be merged. | - |
| `federate.use_diff` | (bool) False | If `True`, the clients would return the variation in local training (i.e., $\delta$) instead of the updated models to the server for federated aggregation. | - | 
| `federate.merge_test_data` | (bool) False | If `True`, clients' test data would be merged and perform global evaluation for efficient simulation. | - |
| `federate.method` | (string) 'FedAvg' | The method used for federated aggregation. | We support existing federated aggregation algorithms (such as 'FedAvg/FedOpt'), 'global' (centralized training), 'local' (isolated training), personalized algorithms ('Ditto/pFedMe/FedEM'), and allow developer to customize. | 
| `federate.ignore_weight` | (bool) False | If `True`, the model updates would be averaged in federated aggregation. | - |
| `federate.use_ss` | (bool) False | If `True`, additively secret sharing would be applied in the FL course. | Only used in vanilla FedAvg in this version. | 
| `federate.restore_from` | (string) '' | The checkpoint file to restore the model. | - |
| `federate.save_to` | (string) '' | The path to save the model. | - | 
| `federate.join_in_info` | (list of string) [] | The information requirements (from server) for joining in the FL course. | We support 'num_sample/client_resource' and allow user customization.
| `federate.sampler` | (string) 'uniform' </br> Choices: {'uniform', 'group'} | The sample strategy of server used for client selection in a training round. | - |
| `federate.` </br>`resource_info_file` | (string) '' | the device information file to record computation and communication ability | - | 
#### `distribute`: for distribute mode
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `distribute.use` | (bool) False | Whether to run FL courses with distribute mode. | If `False`, all the related configurations (`cfg.distribute.xxx`) would not take effect.  |
| `distribute.server_host` | (string) '0.0.0.0' | The host of server's ip address for communication | - |
| `distribute.server_port` | (string) 50050 | The port of server's ip address for communication | - |
| `distribute.client_host` | (string) '0.0.0.0' | The host of client's ip address for communication | - |
| `distribute.client_port` | (string) 50050 | The port of client's ip address for communication | - |
| `distribute.role` | (string) 'client' </br> Choices: {'server', 'client'} | The role of the worker | - |
| `distribute.data_file` | (string) 'data' | The path to the data dile | - |
| `distribute.data_idx` | (int) -1 | It is used to specify the data index in distributed mode when adopting a centralized dataset for simulation (formatted as {data_idx: data/dataloader}). | `data_idx=-1` means that the entire dataset is owned by the participant. And we randomly sample the index in simulation for other invalid values excepted for -1.
| `distribute.` </br>`grpc_max_send_message_length` | (int) 100 * 1024 * 1024 | The maximum length of sent messages | - |
| `distribute.` </br>`grpc_max_receive_message_length` | (int) 100 * 1024 * 1024 | The maximum length of received messages | - |
| `distribute.`grpc_enable_http_proxy | (bool) False | Whether to enable http proxy | - |
#### `vertical`: for vertical federated learning
| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `vertical.use` | (bool) False | Whether to run vertical FL. | If `False`, all the related configurations (`cfg.vertical.xxx`) would not take effect.  |
| `vertical.encryption` | (string) `paillier` | The encryption algorithms used in vertical FL. | - |
| `vertical.dims` | (list of int) [5,10] | The dimensions of the input features for participants. | - |
| `vertical.key_size` | (int) 3072 | The length (bit) of the public keys. | - | 

### Asynchronous Training Strategies
The configurations related to applying asynchronous training strategies in FL are defined in `cfg_asyn.py`.

| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| `asyn.use` | (bool) False | Whether to use asynchronous training strategies. | If `False`, all the related configurations (`cfg.asyn.xxx`) would not take effect.  |
| `asyn.time_budget` | (int/float) 0 | The predefined time budget (seconds) for each training round. | `time_budget`<=0 means the time budget is not applied. |
| `asyn.min_received_num` | (int) 2 | The minimal number of received feedback for the server to trigger federated aggregation. | - |
| `asyn.min_received_rate` | (float) -1.0 | The minimal ratio of received feedback w.r.t. the sampled clients for the server to trigger federated aggregation. | - |
| `asyn.staleness_toleration` | (int) 0 | The threshold of the tolerable staleness in federated aggregation. | - | 
| `asyn.` </br>`staleness_discount_factor` | (float) 1.0 | The discount factor for the staled feedback in federated aggregation. | - |
| `asyn.aggregator` | (string) 'goal_achieved' </br> Choices: {'goal_achieved', 'time_up'} | The condition for federated aggregation. | 'goal_achieved': perform aggregation when the defined number of feedback has been received; 'time_up': perform aggregation when the allocated time budget has been run out. |
| `asyn.broadcast_manner` | (string) 'after_aggregating' </br> Choices: {'after_aggregating', 'after_receiving'} | The broadcasting manner of server. | 'after_aggregating': broadcast the up-to-date global model after performing federated aggregation; 'after_receiving': broadcast the up-to-date global model after receiving the model update from clients. |
| `asyn.overselection` | (bool) False | Whether to use the overselection technique | - |

## Configurations
We summarize all the customizable configurations here.

### Data
The configurations related to the data/dataset are defined in `cfg_data.py`.

| Name |  (Type) Default Value | Description | Note |
|:----:|:-----:|:---------- |:---- |
| data.root | (string) 'data' | <font size=1> The folder where the data file located. `data.root` would be used together with `data.type` to load the dataset. </font> | - |


### Asynchronous Training Strategies
The configurations related to applying asynchronous training strategies in FL are defined in `cfg_asyn.py`.

| Name |  (Type) Default Value |  Description  | Note |
|:----:|:-----:|:---------- |:---- |
| asyn.use | (bool) False| Whether to use asynchronous training strategies. | If `False`, all the related configurations (`cfg.asyn.xxx`) would not task effect.  | - |
| asyn.time_budget | (int/float) 0 | The predefined time budget (seconds) for each training round. | `time_budget`<=0 means the time budget is not applied. |
| asyn.min_received_num | (int) 2 | The minimal number of received feedback for the server to trigger federated aggregation. | - |
| asyn.min_received_rate | (float) -1.0 | The minimal ratio of received feedback w.r.t. the sampled clients for the server to trigger federated aggregation. | - |
| asyn.staleness_toleration | (int) 0 | The threshold of the tolerable staleness in federated aggregation. | - | 
| asyn.staleness_discount_factor | (float) 1.0 | The discount factor for the staled feedback in federated aggregation. | - |
| asyn.aggregator | (string) 'goal_achieved' </br> Choices: {'goal_achieved', 'time_up'} | The condition for federated aggregation. | 'goal_achieved': perform aggregation when the defined number of feedback has been received; 'time_up': perform aggregation when the allocated time budget has been run out. |
| asyn.broadcast_manner | (string) 'after_aggregating' </br> Choices: {'after_aggregating', 'after_receiving'} | The broadcasting manner of server. | 'after_aggregating': broadcast the up-to-date global model after performing federated aggregation; 'after_receiving': broadcast the up-to-date global model after receiving the model update from clients. |
| asyn.overselection | (bool) False | Whether to use the overselection technique | - |

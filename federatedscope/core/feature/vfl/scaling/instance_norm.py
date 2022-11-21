import numpy as np

from federatedscope.core.message import Message


def wrap_instance_norm_for_server(worker):
    """
    This function is to perform instance norm vfl tabular data for server.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl server with instance norm.
    """
    def trigger_for_start(self):
        if self.check_client_join_in():
            self.broadcast_client_address()
            # Ask for instance_mean
            self.msg_buffer['instance_mean'] = {}
            self.comm_manager.send(
                Message(msg_type='ask_for_instance_mean',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp))
            # self.broadcast_model_para()

    def callback_func_for_instance_mean(self, message: Message):
        stat = message.content

        self.msg_buffer['instance_mean']['sender'] = stat

        if len(self.msg_buffer['instance_mean'].keys()) == self._client_num:
            # Aggregate mean
            instance_mean = {}
            for split in self.msg_buffer['instance_mean']['sender']:
                dim = 0
                feat_sum = np.zeros_like(
                    self.msg_buffer['instance_mean']['sender'][split]['mean'])
                for stat in self.msg_buffer['instance_mean'].values():
                    feat_sum += stat[split]['mean'] * stat[split]['dim']
                    dim += stat[split]['dim']
                instance_mean[split] = feat_sum / dim
            self.comm_manager.send(
                Message(msg_type='instance_mean',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        timestamp=self.cur_timestamp,
                        content=instance_mean))

    def callback_func_for_instance_var(self, message: Message):
        # Aggregate std and mean
        ...

    return worker


def wrap_instance_norm_for_client(worker):
    """
    This function is to perform instance norm vfl tabular data for client.
    Args:
        worker: ``federatedscope.core.workers.Worker`` to be wrapped

    Returns:
        Wrap vfl client with instance norm.
    """
    def callback_func_for_ask_for_mean(self, message: Message):
        sender = message.sender
        stat = {}
        # Calculate mean
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(worker.data, split)
                if split_data is not None and 'x' in split_data:
                    stat[split] = {
                        'mean': np.mean(split_data['x'], axis=1),
                        'dim': np.shape(split_data['x'])[1],
                    }

        # Send mean
        self.comm_manager.send(
            Message(msg_type='instance_mean',
                    sender=self.ID,
                    receiver=sender,
                    timestamp=self.cur_timestamp,
                    content=stat))

    def callback_func_for_instance_mean(self, message: Message):
        sender = message.sender
        feat_mean = message.content
        stat = {}
        # Send std
        for split in ['train_data', 'val_data', 'test_data']:
            if hasattr(self.data, split):
                split_data = getattr(worker.data, split)
                if split_data is not None and 'x' in split_data:
                    stat[split] = {
                        'std': np.mean((split_data['x'].T - feat_mean).T**2,
                                       axis=1),
                        'dim': np.shape(split_data['x'])[1],
                    }

        # Send mean
        self.comm_manager.send(
            Message(msg_type='instance_std',
                    sender=self.ID,
                    receiver=sender,
                    timestamp=self.cur_timestamp,
                    content=stat))

    def callback_func_for_instance_norm(self, message: Message):
        # Send start
        ...

    return worker

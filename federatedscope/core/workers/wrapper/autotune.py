import copy
import types
import logging

from federatedscope.core.message import Message
from federatedscope.autotune.utils import flatten_dict, config2cmdargs

logger = logging.getLogger(__name__)


def wrap_autotune_server(server):
    tmp_trigger_for_train = server.trigger_for_train

    def trigger_for_train(self,
                          trigger_train_func,
                          kwargs_for_trigger_train_func={}):
        cfg = copy.deepcopy(self._cfg)
        cfg.defrost()
        cfg.clear_aux_info()
        del cfg['distribute']
        cfg = config2cmdargs(flatten_dict(cfg))

        # broadcast cfg
        self.comm_manager.send(
            Message(msg_type='cfg',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=cfg))
        tmp_trigger_for_train(trigger_train_func,
                              kwargs_for_trigger_train_func)

    # Bind method to instance
    server.trigger_for_train = types.MethodType(trigger_for_train, server)

    return server


def wrap_autotune_client(client):
    def callback_funcs_for_cfg(self, message: Message):
        sender = message.sender
        new_cfg = message.content

        if sender == self.server_id and self._cfg.hpo.use:
            logger.info("Receive a new `cfg`, and start to reinitialize.")
            self._cfg.defrost()
            # TODO: Some var might remain unchanged
            self._cfg.merge_from_list(new_cfg)
            self._cfg.freeze()

    # Bind method to instance
    client.callback_funcs_for_cfg = types.MethodType(callback_funcs_for_cfg,
                                                     client)

    # Register handlers functions
    client.register_handlers('cfg', client.callback_funcs_for_cfg)

    return client

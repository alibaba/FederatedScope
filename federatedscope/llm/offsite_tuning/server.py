import logging

from federatedscope.core.message import Message, b64serializer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_emulator_and_adapter

logger = logging.getLogger(__name__)


class OffsiteTuningServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(OffsiteTuningServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

    # TODO: Rename the `trigger_for_feat_engr` to `before_start_train`
    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        compress_strategy = self._cfg.llm.offsite_tuning.strategy

        # TODO: move to __init__ and replace self.model
        adapter_model = \
            generate_emulator_and_adapter(self.model,
                                          strategy=compress_strategy)
        # Convert to byte
        logger.info('Server: Converting emulator and adapter.')
        emulator_and_adapter = b64serializer(adapter_model)

        # Send
        self.comm_manager.send(
            Message(msg_type='emulator_and_adapter',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    timestamp=self.cur_timestamp,
                    content=emulator_and_adapter))

        trigger_train_func(**kwargs_for_trigger_train_func)

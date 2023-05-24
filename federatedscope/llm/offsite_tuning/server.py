from federatedscope.core.message import Message, b64serializer
from federatedscope.core.workers.server import Server

from federatedscope.llm.offsite_tuning.utils import \
    generate_emulator_and_adapter


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

    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}):
        compress_strategy = self.cfg.llm.offsiet_tuning.strategy

        # TODO: move to __init__ and replace self.model
        adapter_model = \
            generate_emulator_and_adapter(self.model,
                                          strategy=compress_strategy)
        # Convert to byte
        emulator_and_adapter = b64serializer(adapter_model)

        # Send
        self.comm_manager.send(
            Message(msg_type='emulator_and_adapter',
                    sender=self.ID,
                    receiver=list(self.comm_manager.get_neighbors().keys()),
                    timestamp=self.cur_timestamp,
                    content=emulator_and_adapter))

        trigger_train_func(**kwargs_for_trigger_train_func)

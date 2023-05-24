import logging

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.auxiliaries.utils import deb64serializer

logger = logging.getLogger(__name__)


class OffsiteTuningClient(Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):
        super(OffsiteTuningClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

    def _register_default_handlers(self):
        super(OffsiteTuningClient, self)._register_default_handlers()
        self.register_handlers('emulator_and_adapter',
                               self.callback_funcs_for_emulator_and_adapter,
                               [None])

    def callback_funcs_for_emulator_and_adapter(self, message: Message):
        # Init model and trainer and start first round

        self.state, content = message.state, message.content

        logger.info(f'Client {self.ID}: Converting emulator and adapter.')
        adapter_model = deb64serializer(content)

        # TODO: Init model & trainer
        ...

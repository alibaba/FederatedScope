from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client


# Build your worker here.
class MyServer(Server):
    def _register_default_handlers(self):
        self.register_handlers('join_in', self.callback_funcs_for_join_in,
                               ['assign_client_id', 'address', 'model_para'])
        self.register_handlers('join_in_info', self.callback_funcs_for_join_in,
                               ['address', 'model_para'])
        self.register_handlers('model_para', self.callback_funcs_model_para,
                               ['model_para', 'evaluate', 'finish'])
        self.register_handlers('metrics', self.callback_funcs_for_metrics,
                               ['converged'])


class MyClient(Client):
    def _register_default_handlers(self):
        self.register_handlers('assign_client_id',
                               self.callback_funcs_for_assign_id, [None])
        self.register_handlers('address', self.callback_funcs_for_address)
        self.register_handlers('model_para',
                               self.callback_funcs_for_model_para,
                               ['model_para', 'ss_model_para'])
        self.register_handlers('evaluate', self.callback_funcs_for_evaluate,
                               ['metrics'])
        self.register_handlers('finish', self.callback_funcs_for_finish)
        self.register_handlers('converged', self.callback_funcs_for_converged)


def call_my_worker(method):
    if method == 'myfedavg':
        worker_builder = {'client': MyClient, 'server': MyServer}
        return worker_builder


register_worker('myfedavg', call_my_worker)

from federatedscope.register import register_worker
from federatedscope.core.workers import Server, Client


# Build your worker here.
class MyClient(Client):
    pass


class MyServer(Server):
    pass


def call_my_worker(method):
    if method == 'mymethod':
        worker_builder = {'client': MyClient, 'server': MyServer}
        return worker_builder


register_worker('mymethod', call_my_worker)

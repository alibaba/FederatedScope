import types
import logging
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


def wrap_client_for_ss_multiplicative(client):
    # TODO: this only works when one of the arguments is a secret piece of
    #  the indicator vector which we do not make it to be a fixed point.
    #  For general cases, we should add a truncation step at the end.
    def ss_multiplicative(self,
                          secret1,
                          secret2,
                          shared_party_num,
                          behavior=None):
        self.secret1 = secret1
        self.secret2 = secret2
        self.behavior = behavior
        self.shared_party_num = shared_party_num
        self.pe_dict = dict()
        self.pf_dict = dict()
        self.res = None
        if self.own_label:
            self.comm_manager.send(
                Message(msg_type='random_numbers',
                        sender=self.ID,
                        state=self.state,
                        receiver=[self.server_id],
                        content=(shared_party_num, len(secret2))))

    def callback_fun_for_beaver_triplets(self, message: Message):
        pa, pb, self.pc = message.content
        pe = self.ss.secret_add_lists([self.secret1, -pa])
        pf = self.ss.secret_add_lists([self.secret2, -pb])

        self.pe_dict[self.ID] = pe
        self.pf_dict[self.ID] = pf
        for i in range(self.shared_party_num):
            if i + 1 != self.ID:
                self.comm_manager.send(
                    Message(msg_type='part_e_and_f',
                            sender=self.ID,
                            state=self.state,
                            receiver=[i + 1],
                            content=(pe, pf)))

    def callback_func_for_part_e_and_f(self, message: Message):
        pe, pf = message.content
        self.pe_dict[message.sender] = pe
        self.pf_dict[message.sender] = pf
        if len(self.pe_dict) == self.shared_party_num:
            e = self.ss.secret_add_lists([x for x in self.pe_dict.values()])
            f = self.ss.secret_add_lists([x for x in self.pf_dict.values()])
            self.pe_dict = {}
            self.pf_dict = {}
            t1 = self.ss.secret_ndarray_star_ndarray(f, self.secret1)
            t2 = self.ss.secret_ndarray_star_ndarray(e, self.secret2)
            if not self.own_label:
                self.res = self.ss.secret_add_lists([t1, t2, self.pc])
            else:
                t3 = self.ss.secret_ndarray_star_ndarray(e, f)
                self.res = self.ss.secret_add_lists([-t3, t1, t2, self.pc])
            self.continue_next()

    def continue_next(self):
        if self.behavior == 'left_child':
            self.set_left_child()
        elif self.behavior == 'right_child':
            self.set_right_child()
        elif self.behavior == 'weight':
            self.set_weight()

    client.ss_multiplicative = types.MethodType(ss_multiplicative, client)
    client.continue_next = types.MethodType(continue_next, client)
    client.callback_fun_for_beaver_triplets = types.MethodType(
        callback_fun_for_beaver_triplets, client)
    client.callback_fun_for_part_e_and_f = types.MethodType(
        callback_func_for_part_e_and_f, client)

    client.register_handlers('beaver_triplets',
                             client.callback_fun_for_beaver_triplets)
    client.register_handlers('part_e_and_f',
                             client.callback_fun_for_part_e_and_f)

    return client


def wrap_server_for_ss_multiplicative(server):
    def callback_func_for_random_numbers(self, message: Message):
        shared_party_num, size = message.content
        a_list, b_list, c_list = self.ss.beaver_triple(size)
        for i in range(shared_party_num):
            self.comm_manager.send(
                Message(msg_type='beaver_triplets',
                        sender=self.ID,
                        receiver=[i + 1],
                        state=self.state,
                        content=(a_list[i], b_list[i], c_list[i])))

    server.callback_func_for_random_numbers = types.MethodType(
        callback_func_for_random_numbers, server)
    server.register_handlers('random_numbers',
                             server.callback_func_for_random_numbers)

    return server

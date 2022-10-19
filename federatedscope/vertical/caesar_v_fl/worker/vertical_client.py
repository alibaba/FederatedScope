import numpy as np
import logging

from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.vertical.dataloader.utils import batch_iter
# since we use an abstract Paillier, so we can ss it by our simple ss scheme,
# for the real one, you may also rewrite the ss scheme
from federatedscope.vertical.Paillier import abstract_paillier
# from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.vertical.simple_secret_sharing import AdditiveSecretSharing

logger = logging.getLogger(__name__)


class vFLClient(Client):
    """
    The client class for CAESAR vertical FL, which customizes the handled
    functions. Please refer to the tutorial for more details about the
    implementation algorithm
    Implementation of CAESAR Vertical FL refer to `When Homomorphic Encryption
    Marries Secret Sharing: Secure Large-Scale Sparse Logistic Regression and
    Applications in Risk Control` [Chen, et al., 2021]
    (https://arxiv.org/abs/2008.08753)

    For convenience, we assume that client B has the label.
    """
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

        super(vFLClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)

        cfg_key_size = config.caesar_vertical.key_size
        self.my_public_key, self.my_private_key = \
            abstract_paillier.generate_paillier_keypair(n_length=cfg_key_size)
        self.data = data
        # A has my_para = w_a, B has my_para = w_b
        self.my_para = None
        # w_a = <w_a>_1 + <w_a>_2 and w_b = <w_b>_1 + <w_b>_2
        self.my_part_of_others_para = None
        self.my_part_of_my_para = None
        self.others_public_key = None
        self.my_part_of_my_z = None
        self.my_part_of_others_part_of_my_z = None
        self.my_part_of_my_part_of_others_z = None
        self.batch_index = None
        self.own_label = ('y' in self.data['train'])
        self.y = None
        self.lr = config.train.optimizer.lr
        self.dataloader = batch_iter(self.data['train'],
                                     self._cfg.data.batch_size,
                                     shuffled=True)
        self.total_round_num = None

        self.ss = AdditiveSecretSharing(shared_party_num=2)

        self.register_handlers('model_para', self.callback_func_for_model_para)
        self.register_handlers('public_key_and_para',
                               self.callback_func_for_public_key_and_para)
        self.register_handlers('sample_data',
                               self.callback_func_for_sample_data)
        self.register_handlers('decrypt', self.callback_func_for_decrypt)
        self.register_handlers('encrypted', self.callback_func_for_encrypted)
        self.register_handlers('three_values',
                               self.callback_func_for_three_values)
        self.register_handlers('complicated_comp',
                               self.callback_func_for_a_to_update_para)
        self.register_handlers('para_update',
                               self.callback_func_for_b_to_update_para)
        self.register_handlers('final_step_for_a',
                               self.callback_func_for_final_step_for_a)
        self.register_handlers('para_exchange',
                               self.callback_func_for_para_exchange)

    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    # A and B receive paras, ss them, and save them respectively,

    def callback_func_for_model_para(self, message: Message):
        self.total_round_num, self.my_para = message.content
        self.my_part_of_my_para, others_part_of_my_para = \
            self.ss.secret_split(self.my_para)

        self.comm_manager.send(
            Message(msg_type='public_key_and_para',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(self.my_public_key, others_part_of_my_para)))

    def callback_func_for_public_key_and_para(self, message: Message):
        self.others_public_key, self.my_part_of_others_para = message.content
        if self.own_label:
            self.move_to_the_next_train()

    # start training
    # B samples data
    # B calls encode()

    def move_to_the_next_train(self):
        index, self.input_x, input_y = self.sample_data()
        self.y = input_y
        self.batch_index = index
        self.comm_manager.send(
            Message(msg_type='sample_data',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=self.batch_index))
        self.encode()

    # A samples the same data as B
    # A also calls encode()

    def callback_func_for_sample_data(self, message: Message):
        index = message.content
        self.batch_index = index
        self.input_x = self.sample_data(index=self.batch_index)
        self.encode()

    # both clients do the following
    def encode(self):
        en_para = [
            self.my_public_key.encrypt(x) for x in self.my_part_of_others_para
        ]
        self.comm_manager.send(
            Message(msg_type='encrypted',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=en_para))

    def callback_func_for_encrypted(self, message: Message):
        en_para = message.content

        upgrade_x = self.ss.upgrade(self.input_x)
        self.my_part_of_my_z = self.ss.mod_matmul(upgrade_x,
                                                  self.my_part_of_my_para)

        tmp1 = self.ss.mod_matmul(upgrade_x, en_para)

        self.my_part_of_others_part_of_my_z, en_z = \
            self.ss.secret_split_for_piece_of_ss(tmp1)

        self.comm_manager.send(
            Message(msg_type='decrypt',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=en_z))

    def callback_func_for_decrypt(self, message: Message):
        en_z = message.content
        self.my_part_of_my_part_of_others_z = [
            self.my_private_key.decrypt(x) for x in en_z
        ]
        if not self.own_label:
            self.a_computes()

    # A computes <z>_1 = <z_a>_1 + <<z_a>_2>_1 + <<z_b>_1>_1,
    # <z>_1 squared and <z>_1 cubed
    # A encrypts the above three vectors and sends them to B

    def a_computes(self):
        tmp1 = [
            self.my_part_of_my_z[i] + self.my_part_of_others_part_of_my_z[i] +
            self.my_part_of_my_part_of_others_z[i]
            for i in range(len(self.my_part_of_my_z))
        ]
        tmp1 = self.ss.mod_funs(tmp1)
        tmp1 = [self.my_public_key.encrypt(x) for x in tmp1]

        self.comm_manager.send(
            Message(msg_type='three_values',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=tmp1))

    # B receives [<z>_1]_a, [<z>_1 squared]_a, [<z>_1 cubed]_a
    # B computes <z>_2 = <z_b>_2 + <<z_a>_2>_2 + <<z_b>_1>_2
    #            [z]_a = [<z>_1]_a + <z>_2
    # B gets <h hat>_2
    # B sends [<y hat>_1]_a = [y hat]_a - <y hat>_2 to A
    # B computes [e]_a = [y hat]_a - y
    #            <e>_2 = <y hat>_2 - y
    #            [e]_a = [y hat]_a - y
    #          [g_b]_a = [e]_a * X_b and ss it
    # B keeps <g_b>_2
    # B sends [<g_b>_1]_a to A
    # B sends [<e>_2]_b to A
    def callback_func_for_three_values(self, message: Message):
        tmp1 = message.content
        z2 = [
            self.my_part_of_my_z[i] + self.my_part_of_my_part_of_others_z[i] +
            self.my_part_of_others_part_of_my_z[i]
            for i in range(len(self.my_part_of_my_z))
        ]
        z2 = self.ss.mod_funs(z2)

        za = self.ss.mod_add(tmp1, z2)

        za = self.ss.fixedpoint2float(za)
        za = [round(x) for x in za]

        # Here is the approximation of sigmoid function,
        # different data and different forms may lead to diversity,
        # here linear is enough
        t1 = [self.ss.const_mul_fixedpoint(1 / 4, x) for x in za]
        y_hat_a = [self.ss.const_add_fixedpoint(1 / 2, x) for x in t1]
        upgrade_y = self.ss.upgrade(self.y)

        e_a = self.ss.mod_add(y_hat_a, -upgrade_y)

        y_hat_2, y_hat_1 = self.ss.secret_split_for_piece_of_ss(y_hat_a)
        e_2 = self.ss.mod_add(y_hat_2, -upgrade_y)

        upgrade_x = self.ss.upgrade(self.input_x)
        g_b_a = self.ss.err_mul_data(e_a, upgrade_x)
        g_b_2, g_b_1 = self.ss.secret_split_for_piece_of_ss(g_b_a)

        # user b update w_b_2
        upgrade_lr = self.ss.upgrade(self.lr)
        t1 = [upgrade_lr * x for x in g_b_2]

        self.my_part_of_my_para = self.ss.mod_add(self.my_part_of_my_para,
                                                  [-x for x in t1])

        encrypted_e_2 = [self.my_public_key.encrypt(x) for x in e_2]
        self.comm_manager.send(
            Message(msg_type="complicated_comp",
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(y_hat_1, g_b_1, encrypted_e_2)))

    # A receives [<y hat>_1]_a and [<g_b>_1]_a and [<e>_2]_b
    # A decrypts [<g_b>_1]_a and gets <g_b>_1
    # A decrypts [<y hat>_1]_a and gets <y hat>_1
    # A computes <g_a>_1 = <e>_1 * X_a
    # A computes [<g_a>_2]_b = [<e>_2]_b * X_b and ss it
    # A gets <<g_a>_2>_1
    # A sends [<<g_a>_2>_2]_b = [<g_a>_2>]_b - <<g_a>_2>_1 to B
    # A updates <w_a>_1 = <w_a>_1 - alpha * (<g_a>_1 + <<g_a>_2>_1)
    # A updates <w_b>_1 = <w_b>_1- alpha * (<g_b>_1)

    def callback_func_for_a_to_update_para(self, message: Message):
        y_hat_1, en_g_b_1, encrypted_e_2 = message.content
        g_b_1 = [self.my_private_key.decrypt(x) for x in en_g_b_1]
        e_1 = [self.my_private_key.decrypt(x) for x in y_hat_1]

        upgrade_x = self.ss.upgrade(self.input_x)
        g_a_1 = self.ss.err_mul_data(e_1, upgrade_x)

        g_a_2 = self.ss.err_mul_data(encrypted_e_2, upgrade_x)
        g_a_2_1, g_a_2_2 = self.ss.secret_split_for_piece_of_ss(g_a_2)

        # user A updates w_a_1
        upgrade_lr = self.ss.upgrade(self.lr)
        t1 = self.ss.mod_funs(
            [g_a_1[i] + g_a_2_1[i] for i in range(len(g_a_1))])
        t2 = [upgrade_lr * x for x in t1]

        self.my_part_of_my_para = self.ss.mod_add(self.my_part_of_my_para,
                                                  [-x for x in t2])

        # user A updates w_b_1
        t3 = [upgrade_lr * x for x in g_b_1]
        self.my_part_of_others_para = self.ss.mod_add(
            self.my_part_of_others_para, [-x for x in t3])

        self.comm_manager.send(
            Message(msg_type="para_update",
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=g_a_2_2))

    # B receives [<<g_a>_2>_2]_b and decrypts it and gets <<g_a>_2>_2
    # B updates <w_a>_2 = <w_a>_2 - alpha * <<g_a>_2>_2
    # B updates <w_b>_2 = <w_b>_2 - alpha * <g_b>_2
    # And goes to the next iteration
    # After training,
    # B sends <w_a>_2 to A

    def callback_func_for_b_to_update_para(self, message: Message):
        g_a_2_2 = message.content
        g_a_2_2 = [self.my_private_key.decrypt(x) for x in g_a_2_2]

        #  user B updates w_a_2
        upgrade_lr = self.ss.upgrade(self.lr)
        t1 = [upgrade_lr * x for x in g_a_2_2]

        self.my_part_of_others_para = self.ss.mod_add(
            self.my_part_of_others_para, [-x for x in t1])

        self.state += 1
        if self.state < self.total_round_num:
            # Move to next round of training
            logger.info(f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
            self.move_to_the_next_train()
        else:
            self.final_step()

    # call exchange_paras(): two clients send each other their part of paras
    def final_step(self):
        self.comm_manager.send(
            Message(msg_type="final_step_for_a",
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=None))
        self.exchange_paras()

    def callback_func_for_final_step_for_a(self, message: Message):
        self.exchange_paras()

    def exchange_paras(self):
        self.comm_manager.send(
            Message(msg_type="para_exchange",
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=self.my_part_of_others_para))

    def callback_func_for_para_exchange(self, message: Message):
        para = message.content
        self.my_para = self.ss.secret_reconstruct_for_ss_pieces(
            (self.my_part_of_my_para, para))
        self.my_para = self.ss.fixedpoint2float(self.my_para)
        self.my_para = self.my_para / self.ss.epsilon

        # print("my_para", self.my_para)
        self.comm_manager.send(
            Message(msg_type='para_for_server',
                    sender=self.ID,
                    receiver=self.server_id,
                    state=self.state,
                    content=self.my_para))

import numpy as np
import logging

from federatedscope.core.workers import Client
from federatedscope.core.message import Message
from federatedscope.vertical.dataloader.utils import batch_iter
# since we use an abstract Paillier, so we can ss it by our simple ss scheme,
# for the real one, you may also rewrite the ss scheme
from federatedscope.vertical.Paillier import abstract_paillier
# from federatedscope.core.secret_sharing import AdditiveSecretSharing
# here we use a simple secret sharing scheme, for different datasets,
# you may rewrite the scheme by using different parameters
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
        self.others_part_of_my_para = None
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
        self.my_part_of_my_para, self.others_part_of_my_para = \
            self.ss.secret_split(self.my_para)
        self.comm_manager.send(
            Message(msg_type='public_key_and_para',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(self.my_public_key, self.others_part_of_my_para)))

    def callback_func_for_public_key_and_para(self, message: Message):
        self.others_public_key, self.my_part_of_others_para = message.content
        if self.own_label:
            self.move_to_the_next_train(None)

    # start training
    # B sample data
    # B call encode()

    def move_to_the_next_train(self, message: Message):
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

    # A sample the same data as B
    # A also call encode()

    def callback_func_for_sample_data(self, message: Message):
        index = message.content
        self.batch_index = index
        self.input_x = self.sample_data(index=self.batch_index)
        self.encode()

    # both clients do the following
    def encode(self):
        tmp = [
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
                    content=tmp))

    def callback_func_for_encrypted(self, message: Message):
        tmp = message.content
        self.my_part_of_my_z = np.matmul(self.input_x, self.my_part_of_my_para)

        tmp1 = np.matmul(self.input_x, tmp)
        self.my_part_of_others_part_of_my_z, tmp = \
            self.ss.secret_split_for_piece_of_ss(tmp1)

        self.comm_manager.send(
            Message(msg_type='decrypt',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=tmp))

    def callback_func_for_decrypt(self, message: Message):
        tmp1 = message.content
        self.my_part_of_my_part_of_others_z = [
            self.my_private_key.decrypt(x) for x in tmp1
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
        tmp2 = [x**2 for x in tmp1]
        tmp3 = [x**3 for x in tmp1]
        tmp1 = [self.my_public_key.encrypt(x) for x in tmp1]
        tmp2 = [self.my_public_key.encrypt(x) for x in tmp2]
        tmp3 = [self.my_public_key.encrypt(x) for x in tmp3]

        self.comm_manager.send(
            Message(msg_type='three_values',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(tmp1, tmp2, tmp3)))

    # B receives [<z>_1]_a, [<z>_1 squared]_a, [<z>_1 cubed]_a
    # B computes <z>_2 = <z_b>_2 + <<z_a>_2>_2 + <<z_b>_1>_2
    #            [z]_a = [<z>_1]_a + <z>_2
    #      [z cubed]_a = [<z>_1 cubed]_a + 3 [<z>_1 squared]_a 0 <z>_2
    #                    + 3 [<z>_1] 0 (<z>_2 squared) + (<z>_2 cubed)
    #        [y hat]_a = q_0 + q_1 * [z]_a + q_2 [z cubed]_a  and ss it
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
        tmp1, tmp2, tmp3 = message.content
        z2 = [
            self.my_part_of_my_z[i] + self.my_part_of_my_part_of_others_z[i] +
            self.my_part_of_others_part_of_my_z[i]
            for i in range(len(self.my_part_of_my_z))
        ]
        za = [tmp1[i] + z2[i] for i in range(len(tmp1))]
        # z3a = [
        #    tmp3[i] + 3 * tmp2[i] * z2[i] + 3 * tmp1[i] * z2[i] * z2[i] +
        #    z2[i] * z2[i] * z2[i] for i in range(len(tmp3))]

        # Here is the approximation of sigmoid function,
        # different data and different forms may lead to diversity,
        # here linear is enough
        y_hat_a = [
            1 / 2 + 1 / 4 * za[i]  # - 1 / 48 * z3a[i]
            for i in range(len(za))
        ]

        e_a = y_hat_a - self.y
        y_hat_2, y_hat_1 = self.ss.secret_split_for_piece_of_ss(y_hat_a)
        e_2 = y_hat_2 - self.y
        g_b_a = np.matmul(e_a, self.input_x)
        g_b_2, g_b_1 = self.ss.secret_split_for_piece_of_ss(g_b_a)
        # user b update w_b_2
        self.my_part_of_my_para = [
            self.my_part_of_my_para[i] - self.lr * g_b_2[i]
            for i in range(len(g_b_2))
        ]
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

        g_a_1 = np.matmul(e_1, self.input_x)
        g_a_2 = np.matmul(encrypted_e_2, self.input_x)
        g_a_2_1, g_a_2_2 = self.ss.secret_split_for_piece_of_ss(g_a_2)
        # user A updates w_a_1
        self.my_part_of_my_para = [
            self.my_part_of_my_para[i] - self.lr * (g_a_1[i] + g_a_2_1[i])
            for i in range(len(g_a_1))
        ]
        # user A updates w_b_1
        self.my_part_of_others_para = [
            self.my_part_of_others_para[i] - self.lr * g_b_1[i]
            for i in range(len(g_b_1))
        ]
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
        tmp = message.content
        tmp = [self.my_private_key.decrypt(x) for x in tmp]
        #  user B updates w_a_2
        self.my_part_of_others_para = [
            self.my_part_of_others_para[i] - self.lr * tmp[i]
            for i in range(len(tmp))
        ]
        self.state += 1
        if self.state < self.total_round_num:
            # Move to next round of training
            logger.info(f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
            # self.final_step() 可用来查看每轮的prediction
            self.move_to_the_next_train(message=None)
        else:
            self.final_step()

    # B sends <w_a>_2 to A

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
        self.my_para = self.ss.secret_reconstruct(
            (self.my_part_of_my_para, para))
        self.comm_manager.send(
            Message(msg_type='para_for_server',
                    sender=self.ID,
                    receiver=self.server_id,
                    state=self.state,
                    content=self.my_para))

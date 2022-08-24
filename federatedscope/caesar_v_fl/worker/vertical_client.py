import numpy as np
import logging

from federatedscope.core.worker import Client
from federatedscope.core.message import Message
from federatedscope.caesar_v_fl.dataloader.utils import batch_iter
from federatedscope.caesar_v_fl.Paillier import abstract_paillier
from federatedscope.core.secret_sharing import AdditiveSecretSharing

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
        self.register_handlers('b_public_key',
                               self.callback_func_for_b_public_key)
        self.register_handlers('a_public_key',
                               self.callback_func_for_a_public_key)
        self.register_handlers('b_is_ready', self.callback_func_for_b_is_ready)
        self.register_handlers('a_is_ready', self.callback_func_for_a_is_ready)
        self.register_handlers('b_first', self.callback_func_for_b_first)
        self.register_handlers('b_part', self.callback_func_for_b_part)
        self.register_handlers('a_part', self.callback_func_for_a_part)
        self.register_handlers('three_values',
                               self.callback_func_for_three_values)
        self.register_handlers('complicated_comp',
                               self.callback_func_for_a_to_update_para)
        self.register_handlers('para_update',
                               self.callback_func_for_b_to_update_para)
        self.register_handlers('final_step_for_a',
                               self.callback_func_for_final_step_for_a)
        self.register_handlers('final_step_for_b',
                               self.callback_func_for_final_step_for_b)

    def ss_scheme(self, vector):
        # v1, v2 = self.ss.secret_split(vector)
        vector = np.array(vector)
        v1 = [np.random.randn() for _ in range(len(vector))]
        v2 = [vector[i] - v1[i] for i in range(len(vector))]
        return v1, v2

    def sample_data(self, index=None):
        if index is None:
            assert self.own_label
            return next(self.dataloader)
        else:
            return self.data['train']['x'][index]

    # A and B receive paras, ss them, and save them respectively,
    # B sends pk_b and w_b_1 to A

    def callback_func_for_model_para(self, message: Message):
        self.total_round_num, self.my_para = message.content
        self.my_part_of_my_para, self.others_part_of_my_para = self.ss_scheme(
            self.my_para)
        if self.own_label:
            self.comm_manager.send(
                Message(msg_type='b_public_key',
                        sender=self.ID,
                        receiver=[
                            each for each in self.comm_manager.neighbors
                            if each != self.server_id
                        ],
                        state=self.state,
                        content=(self.my_public_key,
                                 self.others_part_of_my_para)))

    # A receives pk_b and w_b_1, saves them
    # A sends pk_a and w_a_2 to B

    def callback_func_for_b_public_key(self, message: Message):
        self.others_public_key, self.my_part_of_others_para = message.content
        # print(self.others_public_key)
        self.comm_manager.send(
            Message(msg_type='a_public_key',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(self.my_public_key, self.others_part_of_my_para)))

    # B receives pk_a and w_a_2, save them
    # B sends "b_is_ready" to A

    def callback_func_for_a_public_key(self, message: Message):
        self.others_public_key, self.my_part_of_others_para = message.content
        self.comm_manager.send(
            Message(msg_type='b_is_ready',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=None))

    # A receives "b_is_ready",
    # A sends "a_is_ready" to B

    def callback_func_for_b_is_ready(self, message: Message):
        self.comm_manager.send(
            Message(msg_type='a_is_ready',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=None))

    # start training
    # B sample data
    # B computes <z_b>_2 = X_b * <w_b>_2, saves it
    # B sends batch_index and [<w_a>_2]_b to A

    def callback_func_for_a_is_ready(self, message: Message):
        index, input_x, input_y = self.sample_data()
        self.y = input_y
        self.batch_index = index
        self.my_part_of_my_z = np.matmul(input_x, self.my_part_of_my_para)
        tmp = [
            self.my_public_key.encrypt(x) for x in self.my_part_of_others_para
        ]
        self.comm_manager.send(
            Message(msg_type='b_first',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(self.batch_index, tmp)))

    # A receives batch_index and [<w_a>_2]_b
    # A computes z_a_1 = X_a * w_a_1, saves it
    # A computes [<z_a>_2]_b = X_a * [<w_a>_2]_b, and ss it,
    # A keeps <<z_a>_2>_1
    # A sends [<<z_a>_2>_2]_b = [<z_a>_2]_b - <<z_a>_2>_1 and [<w_b>_1]_a to B

    def callback_func_for_b_first(self, message: Message):
        index, tmp = message.content
        self.batch_index = index
        input_x = self.sample_data(index=self.batch_index)
        self.my_part_of_my_z = np.matmul(input_x, self.my_part_of_my_para)
        tmp1 = np.matmul(input_x, tmp)
        self.my_part_of_others_part_of_my_z, tmp = self.ss_scheme(tmp1)
        encrypted = [
            self.my_public_key.encrypt(x) for x in self.my_part_of_others_para
        ]
        self.comm_manager.send(
            Message(msg_type='b_part',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=(tmp, encrypted)))

    # B receives [<<z_a>_2>_2]_b and [<w_b>_1]_a
    # B decrypts [<<z_a>_2>_2]_b and keeps <<z_a>_2>_2
    # B computes [<z_b>_1]_a = X_b * [<w_b>_1]_a and ss it
    # B keeps <<z_b>_1>_2 and
    # B sends [<<z_b>_1>_1]_a = [<z_b>_1]_a - <<z_b>_1>_2 to A

    def callback_func_for_b_part(self, message: Message):
        tmp1, tmp2 = message.content
        self.my_part_of_my_part_of_others_z = [
            self.my_private_key.decrypt(x) for x in tmp1
        ]
        input_x = self.sample_data(index=self.batch_index)
        tmp3 = np.matmul(input_x, tmp2)
        self.my_part_of_others_part_of_my_z, tmp = self.ss_scheme(tmp3)
        self.comm_manager.send(
            Message(msg_type='a_part',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=tmp))

    # A receives [<<z_b>_1>_1]_a and decrypts it and gets <<z_b>_1>_1
    # A computes <z>_1 = <z_a>_1 + <<z_a>_2>_1 + <<z_b>_1>_1,
    # <z>_1 squared and <z>_1 cubed
    # A encrypts the above three vectors and sends them to B

    def callback_func_for_a_part(self, message: Message):
        self.my_part_of_my_part_of_others_z = [
            self.my_private_key.decrypt(x) for x in message.content
        ]

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
        y_hat_2, y_hat_1 = self.ss_scheme(y_hat_a)
        e_2 = y_hat_2 - self.y
        input_x = self.sample_data(index=self.batch_index)
        g_b_a = np.matmul(e_a, input_x)
        g_b_2, g_b_1 = self.ss_scheme(g_b_a)
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
        input_x = self.sample_data(index=self.batch_index)
        g_a_1 = np.matmul(e_1, input_x)
        g_a_2 = np.matmul(encrypted_e_2, input_x)
        g_a_2_1, g_a_2_2 = self.ss_scheme(g_a_2)
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
        # user B updates w_a_2
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
            self.callback_func_for_a_is_ready(message=None)
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
                    content=self.my_part_of_others_para))

    # A receives <w_a>_2, and computes w_a = <w_a>_1 + <w_a>_2
    # A sends <w_b>_1 to B. A Ends.

    def callback_func_for_final_step_for_a(self, message: Message):
        tmp = message.content
        self.my_para = [
            self.my_part_of_my_para[i] + tmp[i] for i in range(len(tmp))
        ]
        self.comm_manager.send(
            Message(msg_type='final_step_for_b',
                    sender=self.ID,
                    receiver=[
                        each for each in self.comm_manager.neighbors
                        if each != self.server_id
                    ],
                    state=self.state,
                    content=self.my_part_of_others_para))
        # here A sends w_a to server
        self.comm_manager.send(
            Message(msg_type='a_para_for_server',
                    sender=self.ID,
                    receiver=self.server_id,
                    state=self.state,
                    content=self.my_para))

    # B receives <w_b>_1 and computes w_b = <w_b>_2 + <w_b>_1. B Ends.

    def callback_func_for_final_step_for_b(self, message: Message):
        tmp = message.content
        self.my_para = [
            self.my_part_of_my_para[i] + tmp[i] for i in range(len(tmp))
        ]
        # here B sends w_b to server
        self.comm_manager.send(
            Message(msg_type='b_para_for_server',
                    sender=self.ID,
                    receiver=self.server_id,
                    state=self.state,
                    content=self.my_para))

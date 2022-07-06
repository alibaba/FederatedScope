import tensorflow as tf
import numpy as np


class LogisticRegression(object):
    def __init__(self, in_channels, class_num, use_bias=True):

        self.input_x = tf.placeholder(tf.float32, [None, in_channels],
                                      name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name='input_y')

        self.out = self.fc_layer(input_x=self.input_x,
                                 in_channels=in_channels,
                                 class_num=class_num,
                                 use_bias=use_bias)

        with tf.name_scope('loss'):
            self.losses = tf.losses.mean_squared_error(predictions=self.out,
                                                       labels=self.input_y)

        with tf.name_scope('train_op'):
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=0.001)
            self.train_op = self.optimizer.minimize(self.losses)

        self.sess = tf.Session()
        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            with self.sess.as_default():
                tf.global_variables_initializer().run()

    def fc_layer(self, input_x, in_channels, class_num, use_bias=True):
        with tf.name_scope('fc'):
            fc_w = tf.Variable(tf.truncated_normal([in_channels, class_num],
                                                   stddev=0.1),
                               name='weight')
            if use_bias:
                fc_b = tf.Variable(tf.constant(0.0, shape=[
                    class_num,
                ]),
                                   name='bias')
                fc_out = tf.nn.bias_add(tf.matmul(input_x, fc_w), fc_b)
            else:
                fc_out = tf.matmul(input_x, fc_w)

        return fc_out

    def to(self, device):
        pass

    def trainable_variables(self):
        return tf.trainable_variables()

    def state_dict(self):
        with self.graph.as_default():
            with self.sess.as_default():
                model_param = list()
                param_name = list()
                for var in tf.global_variables():
                    param = self.graph.get_tensor_by_name(var.name).eval()
                    if 'weight' in var.name:
                        param = np.transpose(param, (1, 0))
                    model_param.append(param)
                    param_name.append(var.name.split(':')[0].replace("/", '.'))

                model_dict = {k: v for k, v in zip(param_name, model_param)}

        return model_dict

    def load_state_dict(self, model_para, strict=False):
        with self.graph.as_default():
            with self.sess.as_default():
                for name in model_para.keys():
                    new_param = model_para[name]

                    param = self.graph.get_tensor_by_name(
                        name.replace('.', '/') + (':0'))
                    if 'weight' in name:
                        new_param = np.transpose(new_param, (1, 0))
                    tf.assign(param, new_param).eval()

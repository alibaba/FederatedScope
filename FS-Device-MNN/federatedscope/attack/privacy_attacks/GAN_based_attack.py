import torch
import torch.nn as nn
from copy import deepcopy
from federatedscope.attack.auxiliary.utils import get_generator
import matplotlib.pyplot as plt


class GANCRA():
    '''
    The implementation of GAN based class representative attack.
    https://dl.acm.org/doi/abs/10.1145/3133956.3134012

    References:

        Hitaj, Briland, Giuseppe Ateniese, and Fernando Perez-Cruz.
    "Deep models under the GAN: information leakage from collaborative deep
    learning." Proceedings of the 2017 ACM SIGSAC conference on computer
    and communications security. 2017.



        Args:
            - target_label_ind (int): the label index whose representative
            - fl_model (object):
            - device (str or int): the device to run; 'cpu' or the device
            index to select; default: 'cpu'.
            - dataset_name (str): the dataset name; default: None
            - noise_dim (int): the dimension of the noise that fed into the
            generator; default: 100
            - batch_size (int): the number of data generated into training;
            default: 16
            - generator_train_epoch (int): the number of training steps
            when training the generator; default: 10
            - lr (float): the learning rate of the generator training;
            default: 0.001
            - sav_pth (str): the path to save the generated data; default:
            'data/'
            - round_num (int): the FL round that starting the attack;
            default: -1.

    '''
    def __init__(self,
                 target_label_ind,
                 fl_model,
                 device='cpu',
                 dataset_name=None,
                 noise_dim=100,
                 batch_size=16,
                 generator_train_epoch=10,
                 lr=0.001,
                 sav_pth='data/',
                 round_num=-1):

        # get dataset's corresponding generator
        self.generator = get_generator(dataset_name=dataset_name)().to(device)
        self.target_label_ind = target_label_ind

        self.discriminator = deepcopy(fl_model)

        self.generator_loss_fun = nn.CrossEntropyLoss()

        self.generator_train_epoch = generator_train_epoch

        # the dimension of the noise input to generator
        self.noise_dim = noise_dim
        self.batch_size = batch_size

        self.device = device

        # define generator optimizer
        self.generator_optimizer = torch.optim.SGD(
            params=self.generator.parameters(), lr=lr)
        self.sav_pth = sav_pth
        self.round_num = round_num
        self.generator_loss_summary = []

    def update_discriminator(self, model):
        '''
        Copy the model of the server as the discriminator

        Args:
            model (object): the model in the server

        Returns: the discriminator

        '''

        self.discriminator = deepcopy(model)

    def discriminator_loss(self):
        pass

    def generator_loss(self, discriminator_output):
        '''
        Get the generator loss based on the discriminator's output

        Args:
            discriminator_output (Tensor): the discriminator's output;
                size: batch_size * n_class

        Returns: generator_loss

        '''

        self.num_class = discriminator_output.size()[1]
        ideal_results = self.target_label_ind * torch.ones(
            discriminator_output.size()[0], dtype=torch.long)

        # ideal_results[:] =  self.target_label_ind

        return self.generator_loss_fun(discriminator_output,
                                       ideal_results.to(self.device))

    def _gradient_closure(self, noise):
        def closure():
            generated_images = self.generator(noise)
            discriminator_output = self.discriminator(generated_images)
            generator_loss = self.generator_loss(discriminator_output)

            generator_loss.backward()
            return generator_loss

        return closure

    def generator_train(self):

        for _ in range(self.generator_train_epoch):

            self.generator.zero_grad()
            self.generator_optimizer.zero_grad()
            noise = torch.randn(size=(self.batch_size, self.noise_dim)).to(
                torch.device(self.device))
            closure = self._gradient_closure(noise)
            tmp_loss = self.generator_optimizer.step(closure)
            self.generator_loss_summary.append(
                tmp_loss.detach().to('cpu').numpy())

    def generate_fake_data(self, data_num=None):
        if data_num is None:
            data_num = self.batch_size
        noise = torch.randn(size=(data_num, self.noise_dim)).to(
            torch.device(self.device))
        generated_images = self.generator(noise)

        generated_label = torch.zeros(self.batch_size, dtype=torch.long).to(
            torch.device(self.device))
        if self.target_label_ind + 1 > self.num_class - 1:
            generated_label[:] = self.target_label_ind - 1
        else:
            generated_label[:] = self.target_label_ind + 1

        return generated_images.detach(), generated_label.detach()

    def sav_image(self, generated_data):
        ind = min(generated_data.shape[0], 16)

        for i in range(ind):
            plt.subplot(4, 4, i + 1)

            plt.imshow(generated_data[i, 0, :, :] * 127.5 + 127.5, cmap='gray')
            # plt.imshow(generated_data[i, 0, :, :] , cmap='gray')
            # plt.imshow()
            plt.axis('off')

        plt.savefig(self.sav_pth + '/' +
                    'image_round_{}.png'.format(self.round_num))
        plt.close()

    def sav_plot_gan_loss(self):
        plt.plot(self.generator_loss_summary)
        plt.savefig(self.sav_pth + '/' +
                    'generator_loss_round_{}.png'.format(self.round_num))
        plt.close()

    def generate_and_save_images(self):
        '''

        Save the generated data and the generator training loss

        '''

        generated_data, _ = self.generate_fake_data()
        generated_data = generated_data.detach().to('cpu')

        self.sav_image(generated_data)
        self.sav_plot_gan_loss()

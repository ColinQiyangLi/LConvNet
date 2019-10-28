import os
import random
from itertools import cycle

import torch
import torch.nn as nn
import numpy as np
import imageio
import glob

from lconvnet.tasks.gan.models.WGAN import WGAN
from lconvnet.tasks.gan.models.WGAN_GP import WGAN_GP
import pprint
import json
from munch import Munch
from torchvision import datasets, transforms
from PIL import Image

class GANSampler(nn.Module):
    def __init__(self, config, generate_type):
        super().__init__()

        self.config = config

        # Load GAN hyperparameters from GAN training json.
        self.gan_config_json_path = config.gan_config_json_path
        self.gan_config = Munch(json.load(open(self.gan_config_json_path)))
        print("-------- GAN Training Config --------")
        pp = pprint.PrettyPrinter()
        pp.pprint(self.gan_config)
        print("------------------------")

        # Instantiate the GAN model class.
        self.gan = self.instantiate_gan()
        # Load weights.
        self.gan.load()

        # Whether we want to sample real of generated images.
        self.generate_type = generate_type
        assert (
            self.generate_type == "real" or self.generate_type == "generated"
        ), "Must be one of 'generated', or 'real'. "

    def forward(self, batch_size):
        assert batch_size == self.gan_config.batch_size

        if self.generate_type == "generated":
            samples = self.gan.get_generated(batch_size)
        elif self.generate_type == "real":
            samples = self.gan.get_real(batch_size)

        return samples

    def instantiate_gan(self):
        if self.gan_config.gan_type == "WGAN":
            gan = WGAN(self.gan_config)
        elif self.gan_config.gan_type == "WGAN_GP":
            gan = WGAN_GP(self.gan_config)
        else:
            raise Exception("[!] There is no option for " +
                            self.gan_config.gan_type)
        return gan

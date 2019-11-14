import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False 
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

class Generator(nn.Module):
    def __init__(self, embd_size, **kwargs):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''
        self.img_shape = kwargs.get('img_shape', (3,32,32))
        latent_dim = kwargs.get('n_filters', 160)
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        img_shape = kwargs.get('img_shape', (3, 32, 32))
        self.model = nn.Sequential(
            *block(latent_dim + embd_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
    
class Discriminator(nn.Module):
    def __init__(self, embd_size, **kwargs):
        super(Discriminator, self).__init__()
        img_shape = kwargs.get('img_shape', (3, 32, 32))

        self.model = nn.Sequential(
            nn.Linear(embd_size + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, embd_size),
        )

    def forward(self, img, condition_embd):
        d_in = torch.cat((img.view(img.size(0), -1), condition_embd, -1))
        validity = self.model(d_in)
        return validity


class ConditionedGenerativeModel(nn.Module):
    '''
    Interface that your models should implement
    '''
    
    def log(x):
        return torch.log(x + 1e-8)

    def __init__(self, embd_size, **kwargs):
        '''
        :param embd_size: int, dimension of the conditional embedding
        '''
        super(ConditionedGenerativeModel, self).__init__()
        self.G = Generator(embd_size, **kwargs)
        self.D = Discriminator(embd_size, **kwargs)
        self.optimizer_G = torch.optim.Adam(self.G.parameters(), lr=kwargs.get('lr',0.2))
        self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=kwargs.get('lr',0.2))
        self.latent_dim = kwargs.get('nr_filters', 160)
        self.n_classes = kwargs.get('embd_size', 0)
        self.adversarial_loss = torch.nn.MSELoss()

        if self.n_classes == 0:
             ValueError('embd_size is 0!')

        self.adversarial_loss = torch.nn.MSELoss()
        if cuda:
            self.G.cuda()
            self.D.cuda()
            self.adversarial_loss.cuda()


    def forward(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: outputs : dict of ouputs, this can be {"d_loss" : d_loss, "g_loss" : g_loss"} for a gan
        '''
        
        # Get batch size            
        batch_size = imgs.shape[0]
                         
        # Ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
                         
        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(condition_embd.type(LongTensor))
                         
        # Train GENERATOR
        self.optimizer_G.zero_grad()
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, self.n_classes, batch_size)))                         
        gen_imgs = self.G(z, gen_labels)
                         
        # Get GENERATOR's loss
        validity = self.D(gen_imgs, gen_labels)
        g_loss = self.adversarial_loss(validity, valid)
                         
        g_loss.backward()
        self.optimizer_G.step()
                         
        # Train DISCRIMINATOR
        # Loss for real images
        validity_real = self.D(real_imgs, labels)
        d_real_loss = self.adversarial_loss(validity_real, valid)
                         
        # Loss for fake images
        validity_fake = self.D(gen_imgs.detach(), gen_labels)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        self.optimizer_D.step()
                         
        return g_loss + d_loss
        

    def likelihood(self, imgs, condition_embd):
        '''
        :param imgs: torch.FloatTensor bsize * c * h * w
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: likelihoods : torch.FloatTensor of size bSize, likelihoods of the images conditioned on the captions
        '''
        raise NotImplementedError

    def sample(self, condition_embd):
        '''
        :param condition_embd: torch.FloatTensor bsize * embd_size
        :return: imgs : torch.FloatTensor of size n_imgs * c * h * w
        '''
        # Get batch size
        bsize = condition_embd.shape[0]

        # Sample noise
        z = Variable(
            FloatTensor(np.random.normal(0, 1, (bsize, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = condition_embd
        labels = Variable(LongTensor(labels))
        imgs = Variable(FloatTensor(self.G(z, labels)))
        return imgs

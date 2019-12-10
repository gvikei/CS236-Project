"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""

from __future__ import print_function

from tensorboardX import SummaryWriter


import argparse
import os
import numpy as np
import random
import torch
import sys
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn


import torch.optim as optim
from torch import autograd
import torch.utils.data
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# For calculating inception score

def get_inception_score(G, ):
    all_samples = []
    n_samples = 1000
    for i in range(10):
        samples = torch.FloatTensor(n_samples//10, 200, 1, 1).normal_(-1, 1)
        samples = samples.cuda(0)
        samples = autograd.Variable(samples, volatile=True)
        all_samples.append(G(samples).cpu().data.numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32))
    print(all_samples.shape)
    return inception_score(list(all_samples), cuda=True, batch_size=32, resize=True, splits=10)


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    MODE = 'wgan-gp' # Valid options are dcgan, wgan, or wgan-gp
    DIM = 128 # This overfits substantially; you're probably better off with 64
    LAMBDA = 10 # Gradient penalty lambda hyperparameter
    CRITIC_ITERS = 5 # How many critic iterations per generator iteration
    BATCH_SIZE = 64 # Batch size
    ITERS = 200000 # How many generator iterations to train for
    OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (3*32*32)

    cifar_text_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=200, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--embed_size', default=100, type=int, help='embed size')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
    parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
    parser.add_argument('--sample_only', type=bool, default=False, help='If enabled, only generate images without doing training.')

    opt = parser.parse_args()
    print(opt)

    os.makedirs(os.path.join(opt.outf, "models"), exist_ok=True)
    os.makedirs(os.path.join(opt.outf, "tensorboard"), exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(opt.outf, "tensorboard"))

    # specify the gpu id if using only 1 gpu
    if opt.ngpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
        print('Using single GPU')

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    # some hyper parameters
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    num_classes = int(opt.num_classes)
    nc = 3
    sample_only=bool(opt.sample_only)

    # Define the generator and initialize the weights
    if opt.dataset == 'imagenet':
        netG = _netG(ngpu, nz)
    else:
        netG = _netG_CIFAR10(ngpu, nz)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
        netG.cuda()
    print(netG, type(netG))
    print(get_inception_score(netG))





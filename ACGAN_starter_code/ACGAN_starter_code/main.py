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
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder
from embedders import BERTEncoder


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
    parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
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

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # datase t
    if opt.dataset == 'imagenet':
        # folder dataset
        dataset = ImageFolder(
            root=opt.dataroot,
            transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.CenterCrop(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]),
            classes_idx=(10, 20)
        )
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(
            root=opt.dataroot, download=True,
            transform=transforms.Compose([
                transforms.Scale(opt.imageSize),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    else:
        raise NotImplementedError("No such dataset {}".format(opt.dataset))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

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
    print(netG)

    # Define the discriminator and initialize the weights
    if opt.dataset == 'imagenet':
        netD = _netD(ngpu, num_classes)
    else:
        netD = _netD_CIFAR10(ngpu, num_classes)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()

    # tensor placeholders
    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    eval_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    dis_label = torch.FloatTensor(opt.batchSize)
    aux_label = torch.LongTensor(opt.batchSize)
    real_label = 1
    fake_label = 0


    # if using cuda
    if opt.cuda:
        netD.cuda()
        netG.cuda()
        dis_criterion.cuda()
        aux_criterion.cuda()
        input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
        noise, eval_noise = noise.cuda(), eval_noise.cuda()
        cudnn.benchmark = True

    use_cuda = (torch.cuda.is_available()) and (opt.cuda)
    if use_cuda:
        gpu = 0
        print('Using CUDA')

    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        one = one.cuda(gpu)
        mone = mone.cuda(gpu)

    # define variables
    input = Variable(input)
    noise = Variable(noise)
    eval_noise = Variable(eval_noise)
    dis_label = Variable(dis_label)
    aux_label = Variable(aux_label)
    encoder = BERTEncoder()
    # noise for evaluation
    eval_noise_ = np.random.normal(0, 1, (opt.batchSize, nz))
    eval_label = np.random.randint(0, num_classes, opt.batchSize)
    if opt.dataset == 'cifar10':
                captions = [cifar_text_labels[per_label] for per_label in eval_label]
                embedding = encoder(eval_label, captions)
                embedding = embedding.detach().numpy()
    eval_noise_[np.arange(opt.batchSize), :opt.embed_size] = embedding[:, :opt.embed_size]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(opt.batchSize, nz, 1, 1))

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


    def calc_gradient_penalty(netD, real_data, fake_data):
        # print ("real_data: ", real_data.size(), real_data.nelement()/opt.batchSize)

        alpha = torch.rand(opt.batchSize, 1)
        alpha = alpha.expand(opt.batchSize, real_data.nelement()//opt.batchSize).contiguous().view(opt.batchSize, 3, 32, 32)
        alpha = alpha.cuda(gpu) if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if use_cuda:
            interpolates = interpolates.cuda(gpu)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradient_penalty = 0
        for i in range(len(disc_interpolates)):
            gradients = autograd.grad(outputs=disc_interpolates[i], inputs=interpolates,
                                      grad_outputs=torch.ones_like(disc_interpolates[i]).cuda(gpu) if use_cuda else torch.ones_like(
                                          disc_interpolates[i]),
                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)

            gradient_penalty += ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

        return gradient_penalty.mean()

    # For generating samples
    def generate_image(frame, netG):
        fixed_noise_128 = torch.randn(128, 128)
        if use_cuda:
            fixed_noise_128 = fixed_noise_128.cuda(gpu)
        noisev = autograd.Variable(fixed_noise_128, volatile=True)
        samples = netG(noisev)
        samples = samples.view(-1, 3, 32, 32)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.cpu().data.numpy()

        lib.save_images.save_images(samples, './tmp/cifar10/samples_{}.jpg'.format(frame))

    # For calculating inception score
    def get_inception_score(G, ):
        all_samples = []
        for i in range(10):
            samples_100 = torch.randn(100, 128)
            if use_cuda:
                samples_100 = samples_100.cuda(gpu)
            samples_100 = autograd.Variable(samples_100, volatile=True)
            all_samples.append(G(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        return lib.inception_score.get_inception_score(list(all_samples))

    avg_loss_D = 0.0
    avg_loss_G = 0.0
    avg_loss_A = 0.0
    avg_loss_W = 0.0


    if sample_only:
        opt.niter=1

    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, label = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            dis_label.resize_(batch_size).fill_(real_label)
            aux_label.resize_(batch_size).copy_(label)
            dis_output, aux_output = netD(input)

            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()

            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, aux_label)

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            label = np.random.randint(0, num_classes, batch_size)
            if opt.dataset == 'cifar10':
                captions = [cifar_text_labels[per_label] for per_label in label]
                embedding = encoder(label, captions)
                embedding = embedding.detach().numpy()
            noise_ = np.random.normal(0, 1, (batch_size, nz))

            noise_[np.arange(batch_size), :opt.embed_size] = embedding[:, :opt.embed_size]
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
            aux_label.resize_(batch_size).copy_(torch.from_numpy(label))

            fake = netG(noise)
            dis_label.data.fill_(fake_label)
            dis_output, aux_output = netD(fake.detach())

            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()

            # train with gradient penalty (GP)
            gradient_penalty = calc_gradient_penalty(netD, input.data, fake.data)
            gradient_penalty.backward()

            D_G_z1 = dis_output.data.mean()
            Wasserstein_D = errD_real - errD_fake
            errD = ((errD_fake - errD_real).mean() + gradient_penalty).mean()

            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            dis_label.data.fill_(real_label)  # fake labels are real for generator cost
            dis_output, aux_output = netD(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            optimizerG.step()

            # compute the average loss
            curr_iter = epoch * len(dataloader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_W = avg_loss_W * curr_iter
            all_loss_G += errG.item()
            all_loss_D += errD.item()
            all_loss_A += accuracy
            all_loss_W += Wasserstein_D.item()
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)
            avg_loss_W = all_loss_W / (curr_iter + 1)

            print('[%d/%d][%d/%d] Loss_W: %.4f (%.4f) Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                  % (epoch, opt.niter, i, len(dataloader),
                     Wasserstein_D.item(), avg_loss_W, errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))

            batches_done = epoch * len(dataloader) + i
            writer.add_scalar('train/bpd', avg_loss_G / np.log(2), batches_done)
            writer.add_scalar('Wasserstein loss', avg_loss_W, batches_done)
            writer.add_scalar('D loss', avg_loss_D, batches_done)
            writer.add_scalar('G loss', avg_loss_G, batches_done)
            writer.add_scalar('Accuracy', avg_loss_A, batches_done)

            if i % 100 == 0 or opt.sample_only:
                vutils.save_image(
                    real_cpu, '%s/real_samples.png' % opt.outf)
                print('Label for eval = {}'.format(eval_label))
                fake = netG(eval_noise)
                vutils.save_image(
                    fake.data,
                    '%s/fake_samples_epoch_%03d_%d.png' % (opt.outf, epoch, i)
                )

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
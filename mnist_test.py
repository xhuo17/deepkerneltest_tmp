# -*- coding: utf-8 -*-
import os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch
import pickle
from models.base import *
from easydict import EasyDict


def gen_opt(use_cuda=False):
    opt = EasyDict()
    opt.n_epochs = 2000
    opt.batch_size = 100
    opt.lr = 0.0002
    opt.img_size = 32
    opt.channels = 1
    opt.hidden = 100  # 100 for mnist, 300 for cifar10
    opt.n = 100
    opt.dtype = torch.float
    opt.type = 'c'
    if use_cuda:
        opt.gpu = 1
        opt.device = torch.device(f'cuda:{opt.gpu}')
    else:
        opt.device = torch.device(f'cpu')
    opt.N_per = 100
    opt.alpha = 0.05
    opt.N1 = 100
    opt.K = 10
    opt.N_test = 100
    opt.N_f = float(opt.N_test)

    return opt


def gen_dataloader(opt):
    # Configure data loader
    os.makedirs("./data/mnist", exist_ok=True)

    dataloader_FULL = DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=60000,
        shuffle=True,
    )
    # Obtain real MNIST images
    for i, (imgs, labels) in enumerate(dataloader_FULL):
        data_all = imgs.float()
        label_all = labels.float()

    dataloader_FULL_te = DataLoader(
        datasets.MNIST(
            "./data/mnist",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=10000,
        shuffle=True,
    )
    for i, (imgs, labels) in enumerate(dataloader_FULL_te):
        data_all_te = imgs.float()
        label_all_te = labels.float()

    return data_all, label_all, data_all_te, label_all_te


def main(use_cuda):
    # Setup seeds
    os.makedirs("./images", exist_ok=True)
    np.random.seed(819)
    torch.manual_seed(819)
    if use_cuda:
        torch.cuda.manual_seed(819)
        torch.backends.cudnn.deterministic = True

    opt = gen_opt(use_cuda)

    # ep_OPT_final = np.zeros([opt.K])
    # s_OPT_final = np.zeros([opt.K])
    # s0_OPT_final = np.zeros([opt.K])
    Results_final = np.zeros([opt.K])

    data_all, label_all, data_all_te, label_all_te = gen_dataloader(opt)

    model = BaseModel(opt).to(opt.device)

    # TODO:
    # Repeat experiments K times (K = 10) and report average test power (rejection rate)
    for kk in range(opt.K):
        dataloader, Fake_MNIST_tr, Fake_MNIST_te = model.__set_input__(kk, data_all, label_all)
        model.__train_forward__(kk, dataloader, Fake_MNIST_tr)

        Results_final[kk] = model.__evaluate__(data_all, Fake_MNIST_te)

        print("Test Power of Baselines (K times): ")
        print(Results_final)
        print("Average Test Power of Baselines (K times): ")
        print("MMD-D: ", (Results_final.sum() / (kk + 1)))
    print(f"training complete.")
    # np.save('./Results_MNIST_' + str(opt.N1) + '_H1_MMD_D_Baselines', Results_final)


if __name__ == '__main__':
    use_cuda = False
    main(use_cuda)

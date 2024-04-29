import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from models.modules import Featurizer, Discriminator
from utils.utils_HD import *
import pickle
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset


# the base model
class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.dtype = opt.dtype
        self.device = opt.device
        self.N_per = opt.N_per  # permutation times
        self.alpha = opt.alpha  # test threshold
        self.N1 = opt.n  # number of samples in one set
        self.K = opt.K  # number of trails
        self.N = opt.N_test  # number of test sets
        self.N_f = opt.N_f  # number of test sets (float)

        self.criterion = nn.CrossEntropyLoss() if opt.type == 'c' else nn.MSELoss()

        # Naming variables
        self.ep_OPT = np.zeros([opt.K])
        self.s_OPT = np.zeros([opt.K])
        self.s0_OPT = np.zeros([opt.K])
        self.Results = np.zeros([6, opt.K])

        self.featurizer = Featurizer(opt)
        self.discriminator = Discriminator(opt)

    def __init_model__(self):
        del self.featurizer, self.discriminator
        self.featurizer = Featurizer(self.opt).to(self.device)
        self.discriminator = Discriminator(self.opt).to(self.device)

    def __set_input__(self, kk, data_all, label_all):
        torch.manual_seed(kk * 19 + self.N1)
        torch.cuda.manual_seed(kk * 19 + self.N1)
        np.random.seed(seed=1223 * (kk + 10) + self.N1)
        self.__init_model__()
        np.random.seed(seed=728 * (kk + 9) + self.opt.N1)
        train_data, train_label = [], []
        ind_M_all = np.arange(4000)
        self.ind_M_tr = np.random.choice(4000, self.opt.N1, replace=False)
        self.ind_M_te = np.delete(ind_M_all, self.ind_M_tr)
        for i in self.ind_M_tr:
            train_data.append(data_all[i].unsqueeze(0))
            train_label.append(label_all[i])

        X = torch.cat(train_data, 0)
        y = torch.from_numpy(np.array(train_label))
        dataset_tmp = TensorDataset(X, y)

        dataloader = DataLoader(
            dataset_tmp,
            batch_size=self.opt.batch_size,
            shuffle=True,
        )

        # Collect fake MNIST images
        Fake_MNIST = pickle.load(open('./Fake_MNIST_data_EP100_N10000.pckl', 'rb'))
        ind_all = np.arange(4000)
        self.ind_tr = np.random.choice(4000, self.opt.N1, replace=False)
        self.ind_te = np.delete(ind_all, self.ind_tr)
        Fake_MNIST_tr = torch.from_numpy(Fake_MNIST[0][self.ind_tr]).float()
        Fake_MNIST_te = torch.from_numpy(Fake_MNIST[0][self.ind_te]).float()
        # REPLACE above 6 lines with
        # Fake_MNIST_tr = data_all[ind_M_tr_all[N1:]]
        # Fake_MNIST_te = data_all[ind_M_te]
        # for validating type-I error

        self.epsilonOPT = torch.log(MatConvert(np.random.rand(1) * 10 ** (-10), self.device, self.dtype))
        self.epsilonOPT.requires_grad = True
        self.sigmaOPT = MatConvert(np.ones(1) * np.sqrt(2 * 32 * 32), self.device, self.dtype)
        self.sigmaOPT.requires_grad = True
        self.sigma0OPT = MatConvert(np.ones(1) * np.sqrt(0.005), self.device, self.dtype)
        self.sigma0OPT.requires_grad = True

        # Initialize optimizers
        self.optimizer_F = torch.optim.Adam(
            list(self.featurizer.parameters()) + [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT],
            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.opt.lr)

        return dataloader, Fake_MNIST_tr, Fake_MNIST_te

    def __train_forward__(self, kk, dataloader, Fake_MNIST_tr):
        self.featurizer.train()
        self.discriminator.train()

        # Initialize parameters
        for epoch in range(self.opt.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):
                ind = np.random.choice(self.N1, imgs.shape[0], replace=False)
                Fake_imgs = Fake_MNIST_tr[ind]
                # Adversarial ground truths
                valid = torch.ones(imgs.shape[0], 1)
                valid.requires_grad = False
                fake = torch.zeros(imgs.shape[0], 1)
                fake.requires_grad = False
                # Configure input
                real_imgs = imgs
                real_imgs.requires_grad = True
                Fake_imgs = Fake_imgs
                Fake_imgs.requires_grad = True
                X = torch.cat([real_imgs, Fake_imgs], 0)
                Y = torch.cat([valid, fake], 0).squeeze().long()
                # ------------------------------
                #  Train deep network for MMD-D
                # ------------------------------
                self.optimizer_F.zero_grad()
                model_output = self.featurizer(X)
                self.ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
                self.sigma = self.sigmaOPT ** 2
                self.sigma0_u = self.sigma0OPT ** 2
                TEMP = MMDu(model_output, imgs.shape[0], X.view(X.shape[0], -1), self.sigma, self.sigma0_u, self.ep)
                mmd_value_temp = -1 * (TEMP[0])
                mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
                STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
                STAT_u.backward()
                self.optimizer_F.step()
                # ------------------------------------------
                #  Train deep network for C2ST-S and C2ST-L
                # ------------------------------------------
                self.optimizer_D.zero_grad()
                loss_C = self.criterion(self.discriminator(X), Y)
                loss_C.backward()
                self.optimizer_D.step()

                if (epoch + 1) % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [CE loss: %f] [Stat J: %f]"
                        % (epoch, self.opt.n_epochs, i, len(dataloader), loss_C.item(), -STAT_u.item())
                    )

    def __evaluate__(self, data_all, Fake_MNIST_te):
        self.featurizer.eval()
        self.discriminator.eval()
        # Compute test power of MMD-D and baselines
        H_u = np.zeros(self.N)
        T_u = np.zeros(self.N)
        M_u = np.zeros(self.N)
        np.random.seed(1102)
        count_u = 0
        with torch.no_grad():
            for k in range(self.N):
                # Fetch test data
                np.random.seed(seed=1223 * (k + 1) + self.N1)
                ind_M = np.random.choice(len(self.ind_M_te), self.N1, replace=False)
                s1 = data_all[self.ind_M_te[ind_M]]
                np.random.seed(seed=728 * (k + 3) + self.N1)
                ind_F = np.random.choice(len(Fake_MNIST_te), self.N1, replace=False)
                s2 = Variable(torch.tensor(Fake_MNIST_te[ind_F]))
                S = torch.cat([s1.cpu(), s2.cpu()], 0)
                Sv = S.view(2 * self.N1, -1)
                # MMD-D
                h_u, threshold_u, mmd_value_u = TST_MMD_u(self.featurizer(S), self.N_per, self.N1, Sv, self.sigma,
                                                          self.sigma0_u, self.ep, self.alpha, self.device, self.dtype)

                count_u = count_u + h_u
                print("MMD-DK:", count_u)
                H_u[k] = h_u
                T_u[k] = threshold_u
                M_u[k] = mmd_value_u

        # Print test power of MMD-D and baselines
        print("Reject rate_u: ", H_u.sum() / self.N_f)

        return H_u.sum() / self.N_f

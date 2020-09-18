from __future__ import print_function
import os
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import time
import pickle
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from glob import glob
from util import *
from torch.utils.data import DataLoader

from dataloader import CelebDataSet
from simple_test import TestSimple
from transit_test import Transition

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='n batches to wait before logging training status')
parser.add_argument('--test', type=int, default=0,
                    help='type of test: 0 = simple, 1 = transition')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
if device.type == 'cuda': torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if device.type == 'cuda' else {}

totensor = transforms.ToTensor()

train_dataset = CelebDataSet(state='train', data_aug=False, rgb=1)
test_dataset = CelebDataSet(state='test', data_aug=False, rgb=1)

train_loader = DataLoader(train_dataset,
                        batch_size=args.batch_size,
                        num_workers=1,
                        pin_memory=True,
                        shuffle=True)

test_loader = DataLoader(test_dataset,
                        batch_size=args.batch_size,
                        num_workers=1,
                        pin_memory=True,
                        shuffle=False)

# test_dataset = CelebDataSet(state='test', rgb=1)

class VAE(nn.Module):
    def __init__(self, nc, ngf, ndf, latent_variable_size):
        super(VAE, self).__init__()

        self.nc = nc
        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size

        # encoder
        self.e1 = nn.Conv2d(nc, ndf, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(ndf)

        self.e2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf*2)

        self.e3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf*4)

        self.e4 = nn.Conv2d(ndf*4, ndf*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(ndf*8)

        self.e5 = nn.Conv2d(ndf*8, ndf*8, 4, 2, 1)
        self.bn5 = nn.BatchNorm2d(ndf*8)

        self.fc1 = nn.Linear(ndf*8*4*4, latent_variable_size)
        self.fc2 = nn.Linear(ndf*8*4*4, latent_variable_size)

        # decoder
        self.d1 = nn.Linear(latent_variable_size, ngf*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(ngf*8*2, ngf*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(ngf*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(ngf*8, ngf*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(ngf*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(ngf*4, ngf*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(ngf*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(ngf*2, ngf, 3, 1)
        self.bn9 = nn.BatchNorm2d(ngf, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(ngf, nc, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.leakyrelu(self.bn1(self.e1(x)))
        h2 = self.leakyrelu(self.bn2(self.e2(h1)))
        h3 = self.leakyrelu(self.bn3(self.e3(h2)))
        h4 = self.leakyrelu(self.bn4(self.e4(h3)))
        h5 = self.leakyrelu(self.bn5(self.e5(h4)))
        h5 = h5.view(-1, self.ndf*8*4*4)

        return self.fc1(h5), self.fc2(h5)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if device.type == 'cuda':
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, self.ngf*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def get_latent_var(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.nc, self.ndf, self.ngf))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar


model = VAE(nc=3, ngf=128, ndf=128, latent_variable_size=500)
model.to(device)

reconstruction_function = nn.BCELoss()
reconstruction_function.size_average = False
def loss_function(recon_x, x, mu, logvar):
    BCE = reconstruction_function(recon_x, x)

    # https://arxiv.org/abs/1312.6114 (Appendix B)
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, target_img in enumerate(train_loader):
        data = Variable(target_img).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*128),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*128)))
    return train_loss / (len(train_loader)*128)

def test(epoch):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, target_img in enumerate(test_loader):
            data = Variable(target_img).to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()

            ### EDIT HERE
            # must de-normalize images
            first_img = data.data.double()
            second_img = recon_batch.data.double()

            torchvision.utils.save_image(0.5*first_img+0.5, \
                f'../imgs/Epoch_{epoch}_data.jpg', nrow=8, padding=2)
            torchvision.utils.save_image(0.5*second_img+0.5, \
                f'../imgs/Epoch_{epoch}_recon.jpg', nrow=8, padding=2)

        test_loss /= (len(test_loader)*128)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss


def perform_latent_space_arithmatics(items):
    # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()

    with torch.no_grad():
        model.eval()
        data = [im for item in items for im in item]
        data = [totensor(i) for i in data]
        data = torch.stack(data, dim=0)
        data = Variable(data).to(device)
        z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
        it = iter(z.split(1))
        z = zip(it, it, it)
        zs = []
        numsample = 11
        for i,j,k in z:
            for factor in np.linspace(0,1,numsample):
                zs.append((i-j)*factor+k)
        z = torch.cat(zs, 0)
        recon = model.decode(z)

        it1 = iter(data.split(1))
        it2 = [iter(recon.split(1))]*numsample
        result = zip(it1, it1, it1, *it2)
        result = [im for item in result for im in item]

        result = torch.cat(result, 0)
        torchvision.utils.save_image(result.data, '../imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): 
    # input is a list of tuples (a,b) where a column of 
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    print(data.shape)

    with torch.no_grad():
        data = Variable(data).to(device)
        z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
        it = iter(z.split(1))
        z = zip(it, it)
        zs = []
        numsample = 11
        for i,j in z:
            for factor in np.linspace(0,1,numsample):
                zs.append(i+(j-i)*factor)
        z = torch.cat(zs, 0)
        recon = model.decode(z)

        # iterators
        it1 = iter(data.split(1))
        it2 = [iter(recon.split(1))]*numsample
        result = zip(it1, it1, *it2)
        result = [im for item in result for im in item]
        
        # get hr, lr, and the latest sr. good_idx ~ [0, 1, -1]
        # https://stackoverflow.com/a/3179119
        l = numsample+2
        good_idx = []
        for i in range(len(items)):
            good_idx.append(i*l)
            good_idx.append((i*l)+1)
            good_idx.append(l*(i+1)-1)
        
        result = [result[i] for i in good_idx]

        result = torch.cat(result, 0)
        torchvision.utils.save_image(result.data, '../imgs/trans.jpg', nrow=2+1, padding=2)


def rand_faces(num=5):
    load_last_model()

    with torch.no_grad():
        model.eval()
        z = torch.randn(num*num, model.latent_variable_size)
        z = Variable(z).to(device)
        recon = model.decode(z)
        torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)

def load_last_model():
    models = glob('../models/*.pth')
    model_ids = [(int(f.split('_')[2]), f) for f in models]
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    return start_epoch, last_cp

def resume_training():
    start_epoch, _ = load_last_model()

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        train_loss = train(epoch)
        test_loss = test(epoch)
        torch.save(model.state_dict(), \
            f'../models/Epoch_{epoch}_Train_loss_{train_loss:.4f}_Test_loss_{test_loss:.4f}.pth')

def last_model_to_cpu():
    _, last_cp = load_last_model()
    model.cpu()
    torch.save(model.state_dict(), '../models/cpu_'+last_cp.split('/')[-1])


### MAIN
if __name__ == '__main__':

    train_dir = '../data/train'
    if not os.path.exists(train_dir): os.makedirs(train_dir) 
    
    if not any(fname.endswith('.jpg') for fname in os.listdir(train_dir)):
        w = TestSimple(device, istrain=True)
        w.write()

    resume_training()

    # img_dir = '../data/test'
    # if not os.path.exists(img_dir): os.makedirs(img_dir)

    # ### TEST
    # if args.test == 0:
    #     if not any(fname.endswith('.jpg') for fname in os.listdir(img_dir)):
    #         #w = IMGWriter(device=device, isplain_test=True)
    #         w = TestSimple(device)
    #         w.write()

    #     start_epoch, last_cp = load_last_model()
    #     test(start_epoch)
    
    # ### TRANSPOSE
    # if args.test == 1:
    #     t = Transition(device)
    #     images_list = t.return_pair_list()
    #     images_list = images_list[:25]
    #     latent_space_transition(images_list)


    # last_model_to_cpu()
    # load_last_model()
    # rand_faces(10)
    # da = load_pickle(test_loader[0])
    # da = da[:120]
    # it = iter(da)
    # l = zip(it, it, it)
    # # latent_space_transition(l)
    # perform_latent_space_arithmatics(l)
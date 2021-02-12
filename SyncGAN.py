import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import MNISTtools
import numpy as np
import models_mnist

# Dataset
MNISTtools.downloadMNIST(path='MNIST_data', unzip=True)
MNISTtools.downloadMNIST(path='FMNIST_data', unzip=True, fashion=True)
x1_train, y1_train = MNISTtools.loadMNIST(dataset="training", path="MNIST_data")
x1_test, y1_test = MNISTtools.loadMNIST(dataset="testing", path="MNIST_data")
x2_train, y2_train = MNISTtools.loadMNIST(dataset="training", path="FMNIST_data")
x2_test, y2_test = MNISTtools.loadMNIST(dataset="testing", path="FMNIST_data")
x1_train = x1_train.astype(np.float32) / 255.
x2_train = x2_train.astype(np.float32) / 255.
x1_test = x1_test.astype(np.float32) / 255.
x2_test = x2_test.astype(np.float32) / 255.

# Prepare Synchronized Dataset
x1_class = []
x2_class = []
for i in range(10):
    x1_c = x1_train[y1_train == i]
    x2_c = x2_train[y2_train == i]
    x1_class.append(x1_c[:5000])
    x2_class.append(x2_c[:5000])

x1_class = np.asarray(x1_class).reshape([-1,784])
x2_class = np.asarray(x2_class).reshape([-1,784])
print(x1_class.shape, x2_class.shape)

def sync_batch(x1_class, x2_class, bsize):
    batch_id = np.random.choice(x1_class.shape[0], bsize)
    return x1_class[batch_id], x2_class[batch_id]

def async_batch(x1_class, x2_class, bsize):
    batch_id1 = np.random.choice(x1_class.shape[0], bsize)
    batch_id2 = np.random.choice(x2_class.shape[0], bsize)
    return x1_class[batch_id1], x2_class[batch_id2]

# Decide which device we want to run on
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def zero_grad_list(net_list):
    for net in net_list:
        net.zero_grad()

z_dim = 64

netG1 = models_mnist.Generator(z_dim).to(device)
netD1 = models_mnist.Discriminator().to(device)
netG2 = models_mnist.Generator(z_dim).to(device)
netD2 = models_mnist.Discriminator().to(device)
netS = models_mnist.Synchronizer().to(device)

optG1 = optim.Adam(netG1.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD1 = optim.Adam(netD1.parameters(), lr=1e-4, betas=(0.5, 0.999))
optG2 = optim.Adam(netG2.parameters(), lr=2e-4, betas=(0.5, 0.999))
optD2 = optim.Adam(netD2.parameters(), lr=1e-4, betas=(0.5, 0.999))
optS = optim.Adam(netS.parameters(), lr=2e-4, betas=(0.5, 0.999))

model_name = "SyncGAN"
out_folder = "out/" + model_name + "/"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)
save_folder =  "save/" + model_name + "/"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

bsize = 32
for i in range(20001):
    # ========== Get Batch ==========
    # x
    x1_sync, x2_sync = sync_batch(x1_class, x2_class, bsize)
    x1_async, x2_async = async_batch(x1_class, x2_class, bsize)
    x1_batch = np.concatenate([x1_sync, x1_async], 0)
    x2_batch = np.concatenate([x2_sync, x2_async], 0)
    x1_batch = torch.tensor(x1_batch).to(device).view(-1,1,28,28)
    x2_batch = torch.tensor(x2_batch).to(device).view(-1,1,28,28)
    # z
    z_sync = torch.randn(bsize, z_dim, device=device)
    z1_async = torch.randn(bsize, z_dim, device=device)
    z2_async = torch.randn(bsize, z_dim, device=device)
    z1_batch = torch.cat((z_sync, z1_async), 0)
    z2_batch = torch.cat((z_sync, z2_async), 0)
    
    # label
    ones = torch.full((bsize, 1), 1.0, device=device)
    zeros = torch.full((bsize, 1), 0.0, device=device)
    
    # ========== Discriminator and Synchronizer ==========
    zero_grad_list([netG1, netD1, netG2, netD2, netS])
    # D1
    x1_samp = netG1(z1_batch)
    d1_fake = netD1(x1_samp)
    d1_fake_loss = nn.BCELoss()(d1_fake, torch.cat([zeros,zeros],0))
    d1_real = netD1(x1_batch)
    d1_real_loss = nn.BCELoss()(d1_real, torch.cat([ones,ones],0))
    d1_loss = d1_real_loss + d1_fake_loss
    d1_loss.backward()
    optD1.step()
    # D2
    x2_samp = netG2(z2_batch)
    d2_fake = netD2(x2_samp)
    d2_fake_loss = nn.BCELoss()(d2_fake, torch.cat([zeros,zeros],0))
    d2_real = netD2(x2_batch)
    d2_real_loss = nn.BCELoss()(d2_real, torch.cat([ones,ones],0))
    d2_loss = d2_real_loss + d2_fake_loss
    d2_loss.backward()
    optD2.step()
    # Sync
    s_real = netS(x1_batch, x2_batch)
    s_real_loss = nn.BCELoss()(s_real, torch.cat([ones,zeros],0))
    s_real_loss.backward()
    optS.step()
    
    # ========== Generator ==========
    zero_grad_list([netG1, netD1, netG2, netD2, netS])
    # G1
    x1_samp = netG1(z1_batch)
    d1_fake = netD1(x1_samp)
    g1_loss = nn.BCELoss()(d1_fake, torch.cat([ones,ones],0))
    # G2
    x2_samp = netG2(z2_batch)
    d2_fake = netD2(x2_samp)
    g2_loss = nn.BCELoss()(d2_fake, torch.cat([ones,ones],0))
    # Sync
    s_fake = netS(x1_samp, x2_samp)
    s_fake_loss = nn.BCELoss()(s_fake, torch.cat([ones,zeros],0))
    # Step
    total_loss = g1_loss + g2_loss + s_fake_loss
    total_loss.backward()
    optG1.step()
    optG2.step()

    # ========== Result ==========
    if i % 50 == 0:
        print("[Iter %s] G1_loss: %.4f | D1_loss: %.4f || G2_loss: %.4f | D2_loss: %.4f || S_real: %.4f | S_fake: %.4f"\
        %(str(i).zfill(5), g1_loss.item(), d1_loss.mean().item(), g2_loss.mean().item(), d2_loss.item(), s_real_loss.item(), s_fake_loss.item()))

    if i%200 == 0:
        # Output Images
        x1_real = torch.tensor(x1_class.reshape(10,-1,1,28,28)[:,0,:,:,:])
        x2_real = torch.tensor(x2_class.reshape(10,-1,1,28,28)[:,0,:,:,:])
        x1_samp = netG1(z1_batch).detach().cpu()
        x2_samp = netG2(z2_batch).detach().cpu()
        x_fig = torch.cat((x1_real, x2_real, x1_samp[0:10], x2_samp[0:10], \
            x1_samp[10:20], x2_samp[10:20], x1_samp[20:30], x2_samp[20:30]), 0)
        fp = out_folder+str(i).zfill(4)+".jpg"
        vutils.save_image(x_fig, fp, nrow=10, padding=2, normalize=True)

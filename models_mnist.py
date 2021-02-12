import torch
import torch.nn as nn
import torch.nn.functional as F
from padding_same_conv import *

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class Generator(nn.Module):
    def __init__(self, z_dim, ndf=128):
        super(Generator, self).__init__()
        self.ndf = ndf
        self.fc1 = nn.Linear(z_dim, 7*7*ndf*4)
        self.conv1 = Conv2d(ndf*4, ndf*2, 3)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = Conv2d(ndf*2, ndf, 5)
        self.bn2 = nn.BatchNorm2d(ndf)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = Conv2d(ndf, 1, 5)
    
    def forward(self, z):
        h_fc1 = F.relu(self.fc1(z))
        h_conv1 = F.relu(self.conv1(h_fc1.view(-1, self.ndf*4, 7, 7)))
        
        h_conv2 = self.up2(h_conv1)
        h_conv2 = F.relu(self.bn2(self.conv2(h_conv2)))

        h_conv3 = self.up3(h_conv2)
        x_samp = torch.sigmoid(self.conv3(h_conv3))
        return x_samp

class Discriminator(nn.Module):
    def __init__(self, ndf=128):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.conv1 = Conv2d(1, ndf, 5)
        self.conv2 = nn.utils.spectral_norm(Conv2d(ndf, ndf*2, 5, stride=2))
        self.conv3 = nn.utils.spectral_norm(Conv2d(ndf*2, ndf*4, 3, stride=2))
        self.fc4 = nn.Linear(7*7*ndf*4, 1)

    def forward(self, x):
        h_conv1 = F.leaky_relu(self.conv1(x))
        h_conv2 = F.leaky_relu(self.conv2(h_conv1))
        h_conv3 = F.leaky_relu(self.conv3(h_conv2))
        d_logit = self.fc4(h_conv3.view(-1, 7*7*self.ndf*4))
        d_prob = torch.sigmoid(d_logit)
        return d_prob

class Synchronizer(nn.Module):
    def __init__(self, ndf=128):
        super(Synchronizer, self).__init__()
        self.ndf = ndf
        # Modal 1
        self.conv1_m1 = Conv2d(1, ndf, 5, stride=2)
        self.conv2_m1 = Conv2d(ndf, ndf*2, 3, stride=2)
        self.fc3_m1 = nn.Linear(7*7*ndf*2, 256)
        # Modal 2
        self.conv1_m2 = Conv2d(1, ndf, 5, stride=2)
        self.conv2_m2 = Conv2d(ndf, ndf*2, 3, stride=2)
        self.fc3_m2 = nn.Linear(7*7*ndf*2, 256)
        # Modal 1 & 2
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 1)

    def forward(self, x1, x2):
        h_conv1_m1 = F.leaky_relu(self.conv1_m1(x1))
        h_conv2_m1 = F.leaky_relu(self.conv2_m1(h_conv1_m1))
        h3_m1 = F.leaky_relu(self.fc3_m1(h_conv2_m1.view(-1,7*7*self.ndf*2)))

        h_conv1_m2 = F.leaky_relu(self.conv1_m2(x2))
        h_conv2_m2 = F.leaky_relu(self.conv2_m2(h_conv1_m2))
        h3_m2 = F.leaky_relu(self.fc3_m2(h_conv2_m2.view(-1,7*7*self.ndf*2)))

        h3_concat = torch.cat((h3_m1, h3_m2), 1)
        h4 = F.leaky_relu(self.fc4(h3_concat))
        s_logit = self.fc5(h4)
        s_prob = torch.sigmoid(s_logit)
        return s_prob

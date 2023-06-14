#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import sys
sys.path.append('../')
from models.nsf import make_flow
import math

# All of these classes are used by the Probabilistic Hard Attention baseline, refactored from https://github.com/samrudhdhirangrej/Probabilistic-Hard-Attention

class VAE(nn.Module):
    def __init__(self, nz, nf, nh, nsfL):
        super(VAE, self).__init__()

        self.nz = nz
        self.nf = nf
        self.nh = nh

        self.encoder = nn.Sequential()
        self.encoder.add_module('fc1', nn.Conv2d(self.nh,self.nh,1))
        self.encoder.add_module('rl1', nn.LeakyReLU())
        self.encoder.add_module('ln1', nn.LayerNorm((self.nh,1,1)))
        self.encoder.add_module('fc2', nn.Conv2d(self.nh, self.nf+self.nz*2, 1))
        self.encoder[-1].bias.data.fill_(0)

        self.nsf = make_flow(dim = self.nz, featdim = self.nf, K=5, hidden_layer=2, hidden_dim=1, num_flow=nsfL)

        self.decoder = nn.Sequential() 
        self.decoder.add_module('1', nn.ConvTranspose2d(nz, nf, 3))          
        self.decoder.add_module('2', nn.LeakyReLU())
        self.decoder.add_module('3', nn.LayerNorm((nf,3,3)))
        self.decoder.add_module('4', nn.ConvTranspose2d(nf, nf, 3))         
        self.decoder.add_module('5', nn.LeakyReLU())
        self.decoder.add_module('6', nn.LayerNorm((nf,5,5)))
        self.decoder.add_module('7', nn.ConvTranspose2d(nf, nf, 3))        
        self.decoder.add_module('8', nn.LeakyReLU())
        self.decoder.add_module('9', nn.LayerNorm((nf,7,7)))
        self.decoder.add_module('10', nn.ConvTranspose2d(nf, nf, 2, stride=2))
        self.decoder.add_module('11', nn.LeakyReLU())
        self.decoder.add_module('12', nn.LayerNorm((nf,14,14)))
        self.decoder.add_module('13', nn.Conv2d(nf, nf, 3, padding=1, padding_mode='replicate')) 
        self.decoder.add_module('14', nn.LeakyReLU())
        self.decoder.add_module('15', nn.LayerNorm((nf,14,14)))
        self.decoder.add_module('16', nn.Conv2d(nf, nf, 3, padding=1, padding_mode='replicate')) 
        self.decoder.add_module('17', nn.LeakyReLU())
        self.decoder.add_module('18', nn.LayerNorm((nf,14,14)))
        self.decoder.add_module('19', nn.Conv2d(nf, nf, 3, padding=1, padding_mode='replicate')) 
        self.decoder.add_module('20', nn.LeakyReLU())
        self.decoder.add_module('21', nn.LayerNorm((nf,14,14)))
        self.decoder.add_module('22', nn.Conv2d(nf, nf, 3, padding=1, padding_mode='replicate'))            
        self.decoder.add_module('23', nn.LeakyReLU())
        self.decoder.add_module('24', nn.LayerNorm((nf,14,14)))
        self.decoder.add_module('25', nn.Conv2d(nf, nf, 3, padding=1, padding_mode='replicate'))  
                   

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std, - 0.5 * torch.sum(logvar + eps.pow(2) + np.log(2*np.pi), -1)

    def standard_normal_logprob(self, z, mu = None, logvar = None):
        if mu is None:
            return -0.5 * (math.log(2 * math.pi) + z.pow(2))
        else:
            return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2)/logvar.exp())

    def forward(self, hidden, p=1):
        
        # print(self.encoder(hidden).shape)
        feat, muz, logvarz = torch.split(self.encoder(hidden).squeeze(-1).squeeze(-1), [self.nf, self.nz, self.nz], 1)

        feat = feat.repeat(p,1)
        muz = muz.repeat(p,1)
        logvarz = logvarz.repeat(p,1)
        
        noise, lossn = self.reparameterize(muz, logvarz)
        z, lossf = self.nsf(feat, noise)
        lossz = - self.standard_normal_logprob(z)
        outfeat = self.decoder(z[:,:,None,None])
 
        return outfeat, lossn, lossf, lossz

class GlimpseLoc(nn.Module):
    def __init__(self, nf, maxloc, gsz, full=False):
        super(GlimpseLoc, self).__init__()

        self.nf = nf
        self.maxloc = float(maxloc)
        self.fcg = nn.Sequential()
        self.fcg.add_module('1',  nn.Conv2d(3, 64, 16, stride=16))
        self.fcg.add_module('2',  nn.LeakyReLU())
        self.fcg.add_module('3',  nn.BatchNorm2d(64))
        self.fcg.add_module('4',  nn.Conv2d(64, 128, 1))
        self.fcg.add_module('5',  nn.LeakyReLU())
        self.fcg.add_module('6',  nn.BatchNorm2d(128))
        self.fcg.add_module('7',  nn.Conv2d(128, 256, 1))
        self.fcg.add_module('8',  nn.LeakyReLU())
        self.fcg.add_module('9',  nn.BatchNorm2d(256))
        self.fcg.add_module('10',  nn.Conv2d(256, nf, 1))

        self.fcl = nn.Sequential()
        self.fcl.add_module('fc1', nn.Conv2d(2, nf, 1))

        self.final = nn.Sequential()
        self.final.add_module('lr1', nn.LeakyReLU())
        self.final.add_module('bn1', nn.BatchNorm2d(nf))

    def forward(self, g, l):
        l = l.float()*2/self.maxloc - 1
        g = self.fcg(g)
        l = self.fcl(l)
        return self.final(g+l)

    def onlyfcg(self, g):
        return self.fcg(g)

    def onlyfcl(self, l):
        l = l.float()*2/self.maxloc - 1
        return self.fcl(l)

    def onlyfinal(self, g, l):
        return self.final(g+l)


class simpleRNN(nn.Module):
    def __init__(self, nf, nh):
        super(simpleRNN, self).__init__()

        self.nf = nf
        self.nh = nh
        self.fch = nn.Conv2d(nh, nh, 1)
        self.fcg = nn.Conv2d(nf, nh, 1)

        self.final = nn.Sequential()
        self.final.add_module('lr1', nn.LeakyReLU())
        self.final.add_module('bn1', nn.LayerNorm((nh,1,1)))

    def forward(self, g, h):
        h = self.fch(h)
        g = self.fcg(g)
        if g.size(-1) > 1: return self.final((g+h).permute(0,2,3,1).reshape(-1, self.nh, 1, 1)).reshape(-1, g.size(2), g.size(3), self.nh).permute(0,3,1,2).contiguous()
        return self.final(g+h)


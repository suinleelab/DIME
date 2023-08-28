# Borrowed from https://github.com/samrudhdhirangrej/Probabilistic-Hard-Attention/blob/80c925afa4c7f9171b8c7690b2549468ec686531/nsf.py
#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
# sys.path.append('../')
from baseline_models.model_utils import VAE, GlimpseLoc, simpleRNN
# sys.path.append('../../')
from torchmetrics import AUROC

class HardAttention(nn.Module):
    def __init__(self, T, nsfL, nf, nh, nz, classes, gz, imsz, ccebal, training_phase, pretrain_checkpoint=None, return_auc=False):
        super(HardAttention, self).__init__()
        self.auc = AUROC(task='multiclass', num_classes=classes)

        self.T = T
        self.nf = nf
        self.nh = nh
        self.nz = nz
        self.gz = gz
        self.imsz = imsz
        self.ccebal = ccebal
        self.maxloc = self.imsz - self.gz + 1
        self.gstride = gz//2
        self.ggridsz = 14#(self.maxloc - 1)//self.gstride + 1 
        self.classes = classes
        self.p = 20
        self.puttoeval = (training_phase=='second')
        self.training_phase = training_phase
        self.return_auc = return_auc

        self.glimpse = GlimpseLoc(self.nf, self.maxloc, self.gz)

        self.rnn = simpleRNN(self.nf, self.nh)

        self.classifier = nn.Sequential()
        self.classifier.add_module('dp', nn.Dropout2d())
        self.classifier.add_module('cn', nn.Conv2d(self.nh, classes, 1))

        if training_phase=='first':
            self.forward = self.forward_phase_one
            self.sample = self.forward
        else:
            if training_phase=='second':
                self.load_state_dict(torch.load(pretrain_checkpoint, map_location=torch.device('gp'))[0])
                self.ccebal=0
                for p in self.parameters():
                    p.requires_grad = False

            self.vae = VAE(self.nz, self.nf, self.nh, nsfL)
            self.wc = nn.Parameter(torch.zeros(1,1,1,1), requires_grad = True)
            if training_phase=='third':
               self.load_state_dict(torch.load(pretrain_checkpoint, map_location=torch.device('cpu'))[0])

            self.forward = self.forward_phase_two_three

    def locsample(self, out, hidden, pastp, mask, p):

        B = out.size(0)//p

        hidden = self.rnn(out, hidden.repeat(p,1,1,1))     
        labels = torch.softmax(self.classifier(hidden), 1) 

        labels = labels.reshape(p, B, self.classes, self.ggridsz**2)
        
        ent = (- labels * torch.log(labels)).sum(-2)
        
        cent = (- labels * torch.log(pastp[None,:,:,None])).sum(-2)
        kld = (cent - ent).mean(0)

        kld[mask.reshape(B, self.ggridsz**2)] = -100
        locmax = kld.argmax(-1)
        loc = torch.stack([locmax//self.ggridsz, locmax%self.ggridsz],-1) 
        kld = kld.reshape(B, 1, self.ggridsz, self.ggridsz)

        return loc, kld

    def sample(self, x, y):
        B = x.size(0)
        auroc = []
        acc = []

        loc = torch.stack(torch.meshgrid(torch.arange(self.ggridsz), torch.arange(self.ggridsz)),0)[None,...].to(x.device) 
        locfeat = self.glimpse.onlyfcl(loc * self.gstride) 

        hidden = torch.zeros(B, self.nh, 1, 1).to(x.device)

        mask = torch.zeros((B, self.ggridsz, self.ggridsz), dtype=torch.bool).to(x.device)
        x = x.unfold(2,self.gz,self.gstride).unfold(3,self.gz,self.gstride)
        
        label = self.classifier(hidden)[:,:,0,0]

        pastp = torch.softmax(label, -1)

        for niter in range(self.T):
            # if niter==0:
            #     loc = torch.randint(0, self.ggridsz, (B,2)).to(x.device)
            # else:
            outfeat, _, _, _ = self.vae(hidden, self.p)
            out = self.glimpse.onlyfinal(outfeat, locfeat)
            loc, _ = self.locsample(out, hidden, pastp, mask, self.p)

            glimpse = x[range(B),:,loc[range(B),0],loc[range(B),1],:,:]
            mask[range(B), loc[range(B),0], loc[range(B),1]] = 1
            glimpse = self.glimpse(glimpse, loc[:,:,None,None]*self.gstride) 

            hidden = self.rnn(glimpse, hidden) 

            label = self.classifier(hidden)[:,:,0,0]
            # print(f"label shape={label.shape}")
            pastp = torch.softmax(label, -1)
            
            if self.return_auc:
                auroc.append(self.auc(label, y))
                
            label = torch.argmax(pastp,-1)
            acc.append((label==y).float().mean())
        
        if not self.return_auc:
            return None, acc
        else:
            return None, auroc

    def forward_phase_two_three(self, x, y):

        if self.puttoeval:
            self.eval()

        B = x.size(0)
        acc = []
        auroc = []
        lossV, lossC = 0, 0
        
        glimpsefeat = self.glimpse.onlyfcg(x)
        # print(f"glimpsefeat shape={glimpsefeat.shape}")
        loc = torch.stack(torch.meshgrid(torch.arange(self.ggridsz), torch.arange(self.ggridsz)),0)[None,...].to(x.device) 

        # print(f"loc.shape={loc.shape}")
        locfeat = self.glimpse.onlyfcl(loc * self.gstride)
        # print(locfeat.shape)
        glfeat = self.glimpse.onlyfinal(glimpsefeat, locfeat)

        hidden = torch.zeros(B, self.nh, 1, 1).to(x.device)
        mask = torch.zeros((B, self.ggridsz, self.ggridsz), dtype=torch.bool).to(x.device)
        outfeat, lossn, lossf, lossz = self.vae(hidden.detach())
        label = self.classifier(hidden)[:,:,0,0]
        # outloss = 0.5*(F.mse_loss(outfeat, glimpsefeat.detach(), reduction='none') / self.wc.exp() + self.wc)
        # outloss = (outloss.sum(1)[mask==1]).sum()

        # lossV = (lossV
        #         + lossn.sum()
        #         + lossf.sum() 
        #         + lossz.sum()
        #         + outloss
        #         )

        # lossC = lossC + F.cross_entropy(label, y, reduction='sum')

        pastp = torch.softmax(label, 1)

        for niter in range(self.T):
            with torch.no_grad():
                if (self.training_phase=='second'):
                    loc = torch.randint(0, self.ggridsz, (B,2)).to(x.device)
                else:
                    out = self.glimpse.onlyfinal(outfeat, locfeat)
                    loc, _ = self.locsample(out, hidden, pastp, mask, 1)

            mask[range(B), loc[range(B),0], loc[range(B),1]] = 1

            glimpse = glfeat[range(B), :, loc[range(B),0], loc[range(B),1]][:,:,None,None]

            hidden = self.rnn(glimpse, hidden) 

            label = self.classifier(hidden)[:,:,0,0]

            outfeat, lossn, lossf, lossz = self.vae(hidden)
            # print(f"outfeat shape={outfeat.shape}")
            # print(f"glimpsefeat shape={glimpsefeat.shape}")
            outloss = 0.5*(F.mse_loss(outfeat, glimpsefeat.detach(), reduction='none') / self.wc.exp() + self.wc)
            outloss = (outloss.sum(1)[mask==1]).sum()
 
            lossV = (lossV
                    + lossn.sum()
                    + lossf.sum() 
                    + lossz.sum()
                    + outloss
                    )
            # print(label.shape)
            lossC = lossC + F.cross_entropy(label, y, reduction='sum')

            pastp = torch.softmax(label, 1)
            acc.append((torch.argmax(pastp,1)==y).float().mean())
            if self.return_auc:
                auroc.append(self.auc(label, y))

        loss = lossV/self.nz + self.ccebal*lossC

        if not self.return_auc:
            return loss/(self.T*B), acc
        else:
            return loss/(self.T*B), auroc

    def forward_phase_one(self, x, y):
        B = x.size(0)
        acc = []
        auroc = []
        loss = 0

        # print(f"x shape={x.shape}")
        x = x.unfold(2,self.gz,1).unfold(3,self.gz,1)
        # print(f"x shape={x.unfold(2,self.gz,1).shape}")
        hidden = torch.zeros((B, self.nh, 1, 1)).to(x.device)
        for niter in range(self.T):
            loc = torch.randint(0, self.maxloc, (B,2)).to(x.device)
            glimpse = x[range(B),:,loc[range(B),0],loc[range(B),1],:,:]
            glimpse = self.glimpse(glimpse, loc[:,:,None,None])
            hidden = self.rnn(glimpse, hidden) 
            label = self.classifier(hidden)[:,:,0,0]
            loss = loss + F.cross_entropy(label, y, reduction='sum')
            if self.return_auc: 
                auroc.append(self.auc(label, y))
            label = torch.argmax(label,-1)
            acc.append((label==y).float().mean())
        
        if not self.return_auc: 
            return loss/(self.T*B), acc
        else:
            return loss/(self.T*B), auroc


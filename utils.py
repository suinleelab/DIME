import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from torch.distributions import RelaxedOneHotCategorical
import numpy as np
from torch.nn import functional as F
import math


def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    Args:
      append:
      mask_size:
    '''
    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m 
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out

class StaticMaskLayer1d(torch.nn.Module):
    '''
    Mask a fixed set of indices from 1d tabular data.
    
    Args:
      inds: array or tensor of indices to select.
    '''
    def __init__(self, inds):
        super().__init__()
        self.inds = inds
        
    def forward(self, x):
        return x[:, self.inds]

class StaticMaskLayer2d(torch.nn.Module):
    '''
    Mask a fixed set of pixels from 2d image data.
    
    Args:
      mask: mask indicating which parts of the image to remove at a patch level.
      patch_size: size of patches in the mask.
    '''

    def __init__(self, mask, patch_size):
        super().__init__()
        self.patch_size = patch_size
        mask = mask.float()

        # Reshape if necessary.
        if len(mask.shape) == 4:
            assert mask.shape[0] == 1
            assert mask.shape[1] == 1
        elif len(mask.shape) == 3:
            assert mask.shape[0] == 1
            mask = torch.unsqueeze(mask, 0)
        elif len(mask.shape) == 2:
            mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
        else:
            raise ValueError(f'unable to reshape mask with size {mask.shape}')
        assert mask.shape[-1] == mask.shape[-2]

        # Upsample mask.
        if patch_size == 1:
            mask = mask
        elif patch_size > 1:
            mask = torch.nn.Upsample(scale_factor=patch_size)(mask)
        else:
            raise ValueError('patch_size should be int >= 1')
        self.register_buffer('mask', mask)
        self.mask_size = self.mask.shape[2] * self.mask.shape[3]

    def forward(self, x):
        out = x * self.mask
        return out

def generate_uniform_mask(batch_size, num_features):
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()

def accuracy(pred, y):
    assert isinstance(pred, torch.Tensor)
    pred = pred.softmax(dim=1).cpu().data.numpy()
    acc = accuracy_score(y, pred.argmax(axis=1))
    return acc

def auc(pred, y):
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().data.numpy()
    return roc_auc_score(y, pred[:, 1])

def get_entropy(pred):
    pred = pred.softmax(dim=1).cpu().data.numpy()
    log_pred = np.array(list(map(lambda x: np.log(x), pred)))

    entropies = np.sum(pred * log_pred, axis=1)
    return -entropies

def get_confidence(pred):
    pred = pred.softmax(dim=1)
    return torch.max(pred, dim=1).values

def normalize(data, a, b):
    return (b - a) * ((data - np.min(data)) / (np.max(data) - np.min(data))) + a

def selection_with_lamda(cmi, feature_costs=None, lamda=None):
    assert isinstance(cmi, torch.Tensor)
    if lamda is None:
        return torch.argmax(cmi, dim=1)
    else:
        return torch.argmax(cmi - lamda * feature_costs, dim=1)

def selection_without_lamda(cmi, feature_costs=None, lamda=None):
    if feature_costs is not None:
        return torch.argmax(cmi/feature_costs, dim=1)
    
    return torch.argmax(cmi, dim=1)

def selection_without_cost(cmi, feature_costs=None, lamda=None):
    return torch.argmax(cmi, dim=1)

def generate_gaussion_cost(dim):
    x = np.arange(0, 784)
    mean = statistics.mean(x)
    sd = statistics.stdev(x)
    return normalize(norm.pdf(x, mean, sd), 0, 1)

def generate_2d_gaussion_cost(size=28, fwhm = 20, center=None):
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return normalize((np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)).flatten(), 0, 1)

def generate_pixel_based_cost(dataset):
    cost = np.array([])
    for data in dataset:
        if len(cost) == 0:
            cost = data[0]
        else:
            cost += data[0]
    return normalize(cost.detach().numpy(), 0, 1)


class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:
            # TODO this is somewhat untested, but seems like best way to preserve argmax
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()


def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot

class ConcreteMask(nn.Module):
    '''
    For differentiable global feature selection.
    
    Args:
      num_features:
      num_select:
      group_matrix:
      append:
      gamma:
    '''

    def __init__(self, num_features, num_select, group_matrix=None, append=False, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, num_features, dtype=torch.float32))
        self.append = append
        self.gamma = gamma
        if group_matrix is None:
            self.group_matrix = None
        else:
            self.register_buffer('group_matrix', group_matrix.float())

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        if self.group_matrix is not None:
            out = x * (m @ self.group_matrix)
        else:
            out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    
class ConcreteMask2d(nn.Module):
    '''
    For differentiable global feature selection with 2d image data.

    Args:
      width:
      patch_size:
      num_select:
      append:
      gamma:
    '''

    def __init__(self, width, patch_size, num_select, gamma=0.2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_select, width ** 2, dtype=torch.float32))
        self.upsample = torch.nn.Upsample(scale_factor=patch_size)
        self.width = width
        self.patch_size = patch_size
        self.gamma = gamma

    def forward(self, x, temp):
        dist = RelaxedOneHotCategorical(temp, logits=self.logits / self.gamma)
        sample = dist.rsample([len(x)])
        m = sample.max(dim=1).values
        m = self.upsample(m.reshape(-1, 1, self.width, self.width))
        out = x * m
        return out
    
import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical, Categorical


def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.

    Args:
      mask_size: number of features.
      append: whether to append mask to the output.
    '''
    def __init__(self, mask_size, append=True):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, mask):
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class MaskLayer2d(nn.Module):
    '''
    Mask layer for zeroing out 2d image data.

    Args:
      mask_width: width of the mask, or the number of patches.
      patch_size: upsampling factor for the mask, or number of pixels along
        the side of each patch.
      append: whether to append mask to the output.
    '''

    def __init__(self, mask_width, patch_size, append=False):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width ** 2

        # Set up upsampling.
        self.patch_size = patch_size
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError('patch_size should be int >= 1')

    def forward(self, x, mask):
        # Reshape if necessary.
        if len(mask.shape) == 2:
            mask = mask.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(mask.shape) != 4:
            raise ValueError(f'cannot determine how to reshape mask with shape = {mask.shape}')

        # Apply mask.
        mask = self.upsample(mask)
        out = x * mask
        if self.append:
            out = torch.cat([out, mask], dim=1)
        return out


class MaskLayerGrouped(nn.Module):
    '''
    Mask layer for tabular data with feature grouping.

    Args:
      groups: matrix of feature groups, where each row is a group.
      append: whether to append mask to the output.
    '''
    def __init__(self, group_matrix, append=True):
        # Verify group matrix.
        assert torch.all(group_matrix.sum(dim=0) == 1)
        assert torch.all((group_matrix == 0) | (group_matrix == 1))

        # Initialize.
        super().__init__()
        self.register_buffer('group_matrix', group_matrix.float())
        self.append = append
        self.mask_size = len(group_matrix)

    def forward(self, x, m):
        out = x * (m @ self.group_matrix)
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
    '''Generate mask with cardinality chosen uniformly at random.'''
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()


def get_entropy(pred):
    '''Calculate entropy, assuming logit predictions.'''
    return Categorical(logits=pred).entropy()


def get_confidence(pred):
    '''Calculate prediction confidence level, assuming logit predictions.'''
    return torch.max(pred.softmax(dim=1), dim=1).values


def ind_to_onehot(inds, n):
    '''Convert index to one-hot encoding.'''
    onehot = torch.zeros(len(inds), n, dtype=torch.float32, device=inds.device)
    onehot[torch.arange(len(inds)), inds] = 1
    return onehot


def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot

# TODO: starting here, these are for baseline methods.


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

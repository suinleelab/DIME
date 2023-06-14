# Borrowed from https://github.com/samrudhdhirangrej/Probabilistic-Hard-Attention/blob/80c925afa4c7f9171b8c7690b2549468ec686531/nsf.py
import torch
import itertools
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import MultivariateNormal


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    bin_idx =  torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1
    return bin_idx

def unconstrained_RQS(inputs,
                      unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
                      min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    
    logit = lambda p: torch.log(p) - torch.log(1-p)
    logabsdetsigmoid = lambda x: F.logsigmoid(x) + F.logsigmoid(-x)
    logabsdetlogit = lambda x: -torch.log(x) - torch.log(1-x)
    
    logds = logabsdetsigmoid(inputs)
    
    inputs = torch.sigmoid(inputs)
    outputs = inputs
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs, logabsdet = RQS(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )

    outputs = outputs * 0.999998 + 0.000001
    logdl = logabsdetlogit(outputs)
    
    return logit(outputs), logds+logabsdet+logdl

def RQS(inputs,
        unnormalized_widths, unnormalized_heights, unnormalized_derivatives, 
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):
    
    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths[..., 0] = 0
    cumwidths[..., -1] = 1
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights[..., 0] = 0
    cumheights[..., -1] = 1
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = searchsorted(cumwidths, inputs)[..., None]
    
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_delta = input_heights/input_bin_widths
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)

    numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
    outputs = input_cumheights + numerator / denominator

    derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta + input_derivatives * (1 - theta).pow(2))
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet

class NSF_AR(nn.Module):
    """ Neural spline flow, coupling layer, [Durkan et al. 2019] """

    def __init__(self, dim, featdim, K=5, feat_hidden_dim=8, hidden_dim=8, hidden_layer=2):
        super().__init__()
        self.dim = dim
        self.K = K
        out_dim = 3*K-1
        hidden_layer = hidden_layer - 1
        self.hidden_layer = hidden_layer
        
        basemask1 = torch.tril(torch.ones((dim,dim)),-1)[:,None,:].repeat(1,hidden_dim,1).reshape(dim*hidden_dim,dim)
        basemask2 = torch.tril(torch.ones((dim,dim)))[:,None,:].repeat(1,hidden_dim,1).reshape(dim*hidden_dim,dim)
        
        first_mask = torch.zeros((featdim*feat_hidden_dim + dim*hidden_dim, featdim+dim))
        first_mask[:,:featdim] = 1
        first_mask[featdim*feat_hidden_dim:,featdim:] = basemask1
        
        if hidden_layer>0:
            middle_mask = torch.zeros((featdim*feat_hidden_dim + dim*hidden_dim, featdim*feat_hidden_dim + dim*hidden_dim))
            middle_mask[:, :featdim*feat_hidden_dim] = 1
            middle_mask[featdim*feat_hidden_dim:, featdim*feat_hidden_dim:] = basemask2[:,:,None].repeat(1,1,hidden_dim).reshape(dim*hidden_dim,dim*hidden_dim)
        
        last_mask = torch.zeros((dim*out_dim,featdim*feat_hidden_dim + hidden_dim*dim))
        last_mask[:, :featdim*feat_hidden_dim] = 1
        last_mask[:, featdim*feat_hidden_dim:] = torch.flip(basemask2.T,[0,1])[:,None,:].repeat(1,out_dim, 1).reshape(dim*out_dim,hidden_dim*dim)
        
        self.first_weight = nn.Parameter(init.uniform_(torch.rand(featdim*feat_hidden_dim + dim*hidden_dim, featdim+dim), - 0.001, 0.001)*first_mask, requires_grad = True)
        self.first_bias = nn.Parameter(torch.zeros(featdim*feat_hidden_dim + dim*hidden_dim), requires_grad = True)
        self.first_ln = nn.LayerNorm((featdim*feat_hidden_dim + dim*hidden_dim))

        self.middle_ln = nn.ModuleList()
        for h in range(self.hidden_layer):
            self.register_parameter('middle_weight'+str(h), nn.Parameter(init.uniform_(torch.rand(featdim*feat_hidden_dim + dim*hidden_dim, featdim*feat_hidden_dim + dim*hidden_dim), -0.001, 0.001)*middle_mask, requires_grad = True))
            self.register_parameter('middle_bias'+str(h), nn.Parameter(torch.zeros(featdim*feat_hidden_dim + dim*hidden_dim), requires_grad = True))
            self.middle_ln.append(nn.LayerNorm((featdim*feat_hidden_dim + dim*hidden_dim)))

        self.last_weight = nn.Parameter(init.uniform_(torch.rand(dim*out_dim,featdim*feat_hidden_dim + hidden_dim*dim), -0.001, 0.001)*last_mask, requires_grad = True)
        constant = np.log(np.exp(1 - DEFAULT_MIN_DERIVATIVE) - 1)
        self.last_bias = nn.Parameter(torch.ones(dim*out_dim)*constant, requires_grad = True)
   
        self.register_buffer('first_mask', first_mask)
        self.register_buffer('middle_mask', middle_mask)
        self.register_buffer('last_mask', last_mask)


    def forward(self, x, feat):
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0]).to(x.device)
      
        out = torch.cat([feat, x],1)
        out = F.linear(out, self.first_weight*self.first_mask, self.first_bias)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.first_ln(out)
        for h in range(self.hidden_layer):
            out = F.linear(out, self.__getattr__('middle_weight'+str(h))*self.middle_mask, self.__getattr__('middle_bias'+str(h)))
            out = F.leaky_relu(out, negative_slope=0.2)
            out = self.middle_ln[h](out)
        out = F.linear(out, self.last_weight*self.last_mask, self.last_bias)
        out = out.reshape(x.size(0), self.dim, 3*self.K-1)
        W, H, D = torch.chunk(out, 3, -1)
        z, log_det = unconstrained_RQS(x, W, H, D)

        return z, log_det.sum(-1)

    
class ActNorm(nn.Module):
    def __init__(self, dim, featdim, feat_hidden_dim):
        super().__init__()

        self.name = 'actnorm'
        self.param_t = nn.Sequential()
        self.param_t.add_module('fc1', nn.Linear(featdim, feat_hidden_dim*featdim))
        self.param_t.add_module('rl1', nn.ReLU())
        self.param_t.add_module('fc2', nn.Linear(feat_hidden_dim*featdim, dim))

        self.param_s = nn.Sequential()
        self.param_s.add_module('fc1', nn.Linear(featdim, feat_hidden_dim*featdim))
        self.param_s.add_module('rl1', nn.ReLU())
        self.param_s.add_module('fc2', nn.Linear(feat_hidden_dim*featdim, dim))

        self.param_s[0].weight.data.uniform_(-0.001, 0.001)
        self.param_t[0].weight.data.uniform_(-0.001, 0.001)
        self.param_s[-1].weight.data.uniform_(-0.001, 0.001)
        self.param_t[-1].weight.data.uniform_(-0.001, 0.001)
        self.param_t[0].bias.data.zero_()
        self.param_t[-1].bias.data.zero_()
        self.param_s[0].bias.data.zero_()
        self.param_s[-1].bias.data.zero_()

        self.register_buffer('data_dep_init_done', torch.zeros(1))
        self.coeff = nn.Parameter(torch.zeros((1,1)), requires_grad = True)

    def forward(self, x, feat):
        # first batch is used for init
        if self.data_dep_init_done.data == 0:
            with torch.no_grad():
                print('in')
                data = x.detach()
                feature = feat.detach()
                logit = lambda p: torch.log(p) - torch.log(1-p)
    
                self.coeff.data = (2/data.std(dim=0, keepdim=True)).mean().log().reshape(self.coeff.data.shape)
                s = self.coeff.exp() * torch.sigmoid(self.param_s(feature))
                self.param_t[-1].bias.data = (-(data * s).mean(dim=0, keepdim=True)).reshape(self.param_t[-1].bias.data.shape)
                self.data_dep_init_done.data.fill_(1)
            
        s = self.coeff.exp() * torch.sigmoid(self.param_s(feat))
        t = self.param_t(feat)
        z = x * s + t
        log_det = torch.sum(s.log(), dim=-1)
        return z, log_det
    
class Permutation(torch.nn.Module):
   
    def __init__(self, dim):
        super(Permutation, self).__init__()
        
        self.dim = dim
        self.p = [e for e in list(reversed(range(dim)))]
        
    def forward(self, inputs, feat):
        return inputs[:,self.p], 0
    
class NormalizingFlowModel(nn.Module):
    """ A Normalizing Flow Model is a (prior, flow) pair """
    
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)
    
    def forward(self, feat, x):
        m, _ = x.shape
        log_det = torch.zeros(m).to(x.device)
        zs = x
        for flow in self.flows:
            zs, ld = flow.forward(zs, feat)
            log_det += ld
        return zs, -log_det


def make_flow(dim, featdim, K=3, hidden_layer=2, hidden_dim=1, feat_hidden_dim=1, num_flow=1):
    flows = [NSF_AR(dim=dim, featdim=featdim, K=K, feat_hidden_dim=feat_hidden_dim, hidden_layer = hidden_layer, hidden_dim=hidden_dim) for _ in range(num_flow)]
    perm = [Permutation(dim=dim) for _ in flows]
    norms = [ActNorm(dim=dim, featdim=featdim, feat_hidden_dim=feat_hidden_dim) for _ in flows]
    flows = list(itertools.chain(*zip(norms, perm, flows)))
    model = NormalizingFlowModel(flows)
    print('number of parameters={}'.format(sum((p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item() for p in model.parameters())))
    return model
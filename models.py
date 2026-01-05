"""
Flexible implicit representation network models.

Based on:
- Sitzmann et al.: https://arxiv.org/pdf/2006.09661
- Tancik et al.: https://arxiv.org/pdf/2006.10739
- Mehrabian et al.: https://arxiv.org/pdf/2409.09323
"""

import torch
import torch.nn as nn
import numpy as np


class Sinusoidal(nn.Module):
    """Sinusoidal activation layer inspired by Sitzmann et al."""
    
    def __init__(self, input_dim, output_dim, freq=30, freq_first=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.omega = freq
        self.omega_first = freq_first

        self.layer = nn.Linear(input_dim, output_dim, bias=True)

        with torch.no_grad():
            if self.omega_first:
                self.layer.weight.uniform_(-1/self.input_dim, 1/self.output_dim)
            else:
                self.layer.weight.uniform_(-np.sqrt(6/self.input_dim) / self.omega,
                                          np.sqrt(6/self.input_dim) / self.omega)
    
    def forward(self, x):
        return torch.sin(self.omega * self.layer(x))


class Radial(nn.Module):
    """Radial basis function layer inspired by Sitzmann et al."""
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.centers = nn.Parameter(torch.Tensor(output_dim, input_dim))
        nn.init.uniform_(self.centers, -1, 1)  # bounds between -1 and 1
        self.sigs = nn.Parameter(torch.Tensor(output_dim))
        nn.init.constant_(self.sigs, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.output_dim, self.input_dim)
        x = input.unsqueeze(1).expand(size)
        c = self.centers.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigs.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, x):
        return torch.exp(-1 * x**2)


class Fourier(nn.Module):
    """
    Fourier layer using Chebyshev polynomials.
    Inspired by Mehrabian et al.
    """
    
    def __init__(self, input_dim, output_dim, gridsize):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gridsize = gridsize

        self.coeffs = torch.nn.Parameter(torch.randn(2, output_dim, input_dim, gridsize) /
                                                (np.sqrt(input_dim) * np.sqrt(gridsize)))
        self.bias = torch.nn.Parameter(torch.zeros(1, output_dim))

    def forward(self, x):
        k = torch.reshape(torch.arange(1, self.gridsize+1, device=x.device), (1,1,1,self.gridsize))
        x = torch.reshape(x, (x.shape[1],1,x.shape[2],1))

        out = torch.sum(torch.cos(k*x) * self.coeffs[0:1],(-2,-1))
        out += torch.sum(torch.sin(k*x) * self.coeffs[1:2],(-2,-1))
        out += self.bias
        out = torch.reshape(out, (-1, self.output_dim))
        return out


class Flex_Model(nn.Module):
    """
    Flexible implicit neural representation model.
    Supports multiple activation types: sin, relu, tanh, rbf, ffn, fkan.
    """
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, nhidden=3, freq=30, activation='sin'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.nhidden = nhidden
        self.freq = freq
        self.activation = activation

        if activation=='sin':
            self.modellst = nhidden * [Sinusoidal(hidden_dim, hidden_dim, freq=freq, freq_first=False)]
            self.modellst.insert(0, Sinusoidal(input_dim, hidden_dim, freq=freq, freq_first=True))
            finallayer = nn.Linear(hidden_dim, output_dim)
            with torch.no_grad():
                finallayer.weight.uniform_(-np.sqrt(6 / hidden_dim) / freq, np.sqrt(6 / hidden_dim) / freq)
            self.modellst.append(finallayer)

        elif activation in ['relu', 'ffn']:
            if activation == 'ffn':
                # Fourier feature network (inspired by Tancik)
                assert input_dim % 2 == 0, 'for ffn, input dim needs to be even'
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.B = torch.randn((input_dim // 2, 2)).to(device) * 10
            self.modellst = nhidden * [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()]
            self.modellst.insert(0, nn.ReLU())
            self.modellst.insert(0, nn.Linear(input_dim, hidden_dim, bias=True))
            self.modellst.append(nn.Linear(hidden_dim, output_dim, bias=True))

        elif activation=='tanh':
            self.modellst = nhidden * [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.Tanh()]
            self.modellst.insert(0, nn.Tanh())
            self.modellst.insert(0, nn.Linear(input_dim, hidden_dim, bias=True))
            self.modellst.append(nn.Linear(hidden_dim, output_dim, bias=True))

        elif activation=='rbf':
            self.modellst = nhidden * [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()]
            self.modellst.insert(0, Radial(input_dim, hidden_dim))
            self.modellst.append(nn.Linear(hidden_dim, output_dim, bias=True))

        elif activation=='fkan':
            # Fourier Kolmogorov-Arnold Network (inspired by Mehrabian)
            self.gridsize = 100  # the paper found 270 worked well
            self.modellst = nhidden * [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.ReLU()]
            self.modellst.insert(0, nn.LayerNorm(hidden_dim))
            self.modellst.insert(0, Fourier(input_dim, hidden_dim, self.gridsize))
            self.modellst.append(nn.Linear(hidden_dim, output_dim, bias=True))

        else:
            raise ValueError('activation must be in [sin, relu, tanh, rbf, ffn, fkan]')

        self.model = nn.Sequential(*self.modellst)

    def forward(self, x):
        x = x.clone().detach().requires_grad_(True)
        if self.activation=='ffn':
            # Fourier feature mapping (inspired by Tancik)
            x_proj = (2*np.pi*x) @ self.B.T
            x = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        output = self.model(x)
        return output, x

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from time import time
# import matplotlib.pyplot as plt
import numpy as np
import gzip, os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm2d
from torch.nn.functional import relu ,dropout
from torchvision import datasets, transforms
from torchvision import datasets, transforms
from .complexLayers import complex_dropout, ComplexConv2d
# from .efficient import ComplexLinear
# from ..normalization import ComplexLayerNorm2d
from .complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU

class ComplexMaxPool2d(nn.Module):

    def __init__(self,kernel_size, stride= None, padding = 0,
                 dilation = 1, return_indices = False, ceil_mode = False):
        super(ComplexMaxPool2d,self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.maxpool_r = nn.MaxPool2d(kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)
        self.maxpool_i = nn.MaxPool2d(kernel_size = self.kernel_size,
                                stride = self.stride, padding = self.padding,
                                dilation = self.dilation, ceil_mode = self.ceil_mode,
                                return_indices = self.return_indices)

    def forward(self,input):
        input_r = input.real
        input_i = input.imag
        return self.maxpool_r(input_r)+1j*self.maxpool_i(input_i)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ComplexFlatten(nn.Module):
    def forward(self, input):
        input_r = input.real
        input_i = input.imag
        input_r = input_r.view(input_r.size()[0], -1)
        input_i = input_i.view(input_i.size()[0], -1)
        return input_r+1j*input_i

class ComplexSequential(nn.Sequential):
    def forward(self, input):

        for module in self._modules.values():
            input = module(input)
        return input

class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        input_r = input.real
        input_i = input.imag
        return torch.complex(self.fc_r(input_r) - self.fc_i(input_i), self.fc_r(input_i) + self.fc_i(input_r))

# class ComplexLinear(Module):
#
#     def __init__(self, in_features, out_features):
#         super(ComplexLinear, self).__init__()
#         self.fc_r = Linear(in_features, out_features)
#         # self.fc_i = Linear(in_features, out_features)
#
#     def forward(self, input):
#         input_r = input.real
#         input_i = input.imag
#         return torch.complex(self.fc_r(input_r), self.fc_r(input_i))
def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        image_size_h, image_size_w = pair(image_size)
        assert image_size_h % patch_size == 0 and image_size_w % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        add_dimension = 56
        num_patches = (image_size_h // patch_size) * (image_size_w // patch_size)
        num_patches2 = (image_size_h // add_dimension) * (image_size_w // add_dimension)
        patch_dim = channels * patch_size ** 2
        patch_dim2 = channels * add_dimension ** 2

        self.conv = ComplexSequential(
            ComplexConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1),
            ComplexBatchNorm2d(16),
            ComplexReLU(),
            ComplexMaxPool2d(2, stride=2),

            ComplexConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            ComplexBatchNorm2d(32),
            ComplexReLU(),
            ComplexMaxPool2d(2, stride=2),
            ComplexFlatten()
        )

        self.dim = dim
        self.num_classes = num_classes
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w c) p1 p2', p1=patch_size, p2=patch_size),
        #     ComplexConv2d(147, 147),
        #     Rearrange('b (h w c) p1 p2 -> b (h w) (p1 p2 c)', h=7,w=7),
        #     ComplexLinear(patch_dim, dim),
        # )
        #
        # self.to_patch_embedding2 = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w c) p1 p2', p1 = add_dimension, p2 = add_dimension),
        #     ComplexConv2d(48, 48),
        #     Rearrange('b (h w c) p1 p2 -> b (h w) (p1 p2 c)', h=4, w=4),
        #     ComplexLinear(patch_dim2, dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            ComplexLinear(patch_dim, dim),
        )
        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = add_dimension, p2 = add_dimension),
            ComplexLinear(patch_dim2, dim),
        )
        # self.cvlinear = ComplexLinear()

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches +1 , dim))
        self.pos_embedding = nn.Parameter(init_(torch.zeros([1, num_patches + 1, dim], dtype=torch.complex64)))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches +1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()
        self.lifunc = ComplexLinear(dim, num_classes)
        self.ln = nn.LayerNorm(dim)
        self.rang1 = Rearrange('b c (h p1) (w p2) -> (b h w c) p1 p2', p1=patch_size, p2=patch_size)
        self.linear1 = ComplexLinear(1152 * 3, dim)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, num_classes)
        # )
        # self.mlp_head = nn.Sequential(
        #     ComplexLayerNorm1d(8, eps=1e-12, affine=False, track_running_stats=True),
        #     ComplexLinear(dim, num_classes)
        # )

    def forward(self, img):
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        x = self.rang1(img)
        batch_size, n_features, n_features = x.shape
        x = x.reshape(-1, 1, n_features, n_features)
        x = self.conv(x)
        batchsize = img.shape[0]
        rang1 = Rearrange('(b h w c) p -> b (h w) (p c)', b=batchsize, c=3, h=7, w=7)
        x = rang1(x)
        # x = self.rang_conv(x)
        x = self.linear1(x)
        # x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)
        cls_token = torch.complex(cls_tokens,cls_tokens2)
        x = torch.cat((cls_token, x), dim=1)
        # pos_embedding = torch.complex(self.pos_embedding,self.pos_embedding2)
        x += self.pos_embedding[:, :(n + 1)]
        # x2 = self.to_patch_embedding2(img)
        # b, n2, _ = x.shape
        # x2 += self.pos_embedding2[:, :n2]
        # x = torch.cat((x, x2), dim=1)




        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        # xr = self.ln(x.real)
        # xi = self.ln(x.imag)
        # x = torch.complex(xr,xi)
        normfunc = ComplexLayerNorm1d(x.shape[-1], elementwise_affine=False)
        x = normfunc(x)
        lifunc = ComplexLinear(self.dim, self.num_classes)
        x = self.lifunc(x)
        return x

class ComplexLayerNorm1d(nn.LayerNorm):

    def forward(self, input):
        # input = torch.complex(inputr, inputi)
        exponential_average_factor = 0.0


        mean_r = input.real.mean([1])
        mean_i = input.imag.mean([1])
        mean = mean_r + 1j * mean_i

        input = input - mean[:,  None]


        n = input.numel() / input.size(0)
        Crr = 1. / n * input.real.pow(2).sum(dim=[1]) + self.eps
        Cii = 1. / n * input.imag.pow(2).sum(dim=[1]) + self.eps
        Cri = (input.real.mul(input.imag)).mean(dim=[1])

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inputr = Rrr[:,  None] * input.real + Rri[:,  None] * input.imag
        inputi = Rii[:,  None] * input.imag + Rri[:,  None] * input.real

        if self.elementwise_affine:
            # input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
            #         self.bias[None,:,0,None,None]).type(torch.complex64) \
            #         +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
            #         self.bias[None,:,1,None,None]).type(torch.complex64)

            inputr = self.weight[:, None, None] * input.real + self.weight[:, None, None] * input.imag + self.bias[None, :, 0, None, None]
            inputi = self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None, None] * input.imag + self.bias[None, :, 1, None, None]
        # del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inputr+1j*inputi



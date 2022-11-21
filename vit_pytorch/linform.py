import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Parameter, init
from linformer.reversible import ReversibleSequence, SequentialSequence
from .efficient import ComplexLinear
# helper functions
# from ..normalization import ComplexLayerNorm2d
from torch.nn.functional import softmax
from .complexLayers import ComplexDropout

class ComplexLayerNorm2d(nn.LayerNorm):

    def forward(self, input):
        # input = torch.complex(inputr, inputi)
        exponential_average_factor = 0.0


        mean_r = input.real.mean([2])
        mean_i = input.imag.mean([2])
        mean = mean_r + 1j * mean_i

        input = input - mean[:, :, None]


        n = input.numel() / input.size(0)/input.size(1)
        Crr = 1. / n * input.real.pow(2).sum(dim=[2]) + self.eps
        Cii = 1. / n * input.imag.pow(2).sum(dim=[2]) + self.eps
        Cri = (input.real.mul(input.imag)).mean(dim=[2])

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inputr = Rrr[:, :, None] * input.real + Rri[:, :, None] * input.imag
        inputi = Rii[:, :, None] * input.imag + Rri[:, :, None] * input.real

        if self.elementwise_affine:
            # input = (self.weight[None,:,0,None,None]*input.real+self.weight[None,:,2,None,None]*input.imag+\
            #         self.bias[None,:,0,None,None]).type(torch.complex64) \
            #         +1j*(self.weight[None,:,2,None,None]*input.real+self.weight[None,:,1,None,None]*input.imag+\
            #         self.bias[None,:,1,None,None]).type(torch.complex64)

            inputr = self.weight[:, None, None] * input.real + self.weight[:, None, None] * input.imag + self.bias[None, :, 0, None, None]
            inputi = self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None, None] * input.imag + self.bias[None, :, 1, None, None]
        # del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inputr+1j*inputi

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(dim)
        # self.norm = ComplexLayerNorm2d(8, eps=1e-12, affine=False, track_running_stats=True)
        # self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        normfun = ComplexLayerNorm2d(x.shape[-1],elementwise_affine=False)
        # xr = self.ln(x.real)
        # xi = self.ln(x.imag)
        # x = xr+1j*xi
        x = normfun(x)
        return self.fn(x)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.ReLU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0., activation = None, glu = False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        # self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.w1 = ComplexLinear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.cv_dropout = ComplexDropout(dropout)
        # self.dropout = nn.Dropout(dropout)
        # self.w2 = nn.Linear(dim * mult, dim)
        self.w2 = ComplexLinear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            # x = self.act(x)
            x = torch.complex(self.act(x.real),self.act(x.imag))
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            xx = torch.complex(self.act(x.real),self.act(x.imag))
            x = torch.complex(xx.real*v.real-xx.imag*v.imag, xx.real*v.imag+xx.imag*v.real)

        # x = self.dropout(x.real)+1j*self.dropout(x.imag)
        # x = self.cv_dropout(x)

        x = self.w2(x)

        return x

class complex_attn(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        # self.to_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_q = nn.Linear(dim, dim_head * heads)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        # self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.to_k = nn.Linear(dim, kv_dim)
        self.proj_k = nn.Parameter(init_(torch.zeros([seq_len, k])))

        self.share_kv = share_kv
        if not share_kv:
            # self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.to_v = nn.Linear(dim, kv_dim)
            self.proj_v = nn.Parameter(init_(torch.zeros([seq_len, k])))

        self.dropout = ComplexDropout(dropout)

        # self.to_out = nn.Linear(dim_head * heads, dim)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def forward(self,xq,xk,xv, context = None, **kwargs):
        b, n, d, d_h, h, k = *xq.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert kv_len == self.seq_len, f'the sequence length of the key / values must be {self.seq_len} - {kv_len} given'

        queries = self.to_q(xq)

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)

        kv_input = xk if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(xv)

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        merge_key_values = lambda t: t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)
        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots
        # attn = dots.softmax(dim=-1)
        # attn = softmax(dots.real, dim=-1).type(torch.complex64) + 1j * softmax(dots.imag, dim=-1).type(torch.complex64)
        # attn = self.dropout(attn)
        # attn = self.dropout(attn.real)+1j*self.dropout(attn.imag)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        out = self.to_out(out)
        return out


class LinformerSelfAttention(nn.Module):
    def __init__(self, dim, seq_len, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, dropout = 0.):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        # self.to_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_q = ComplexLinear(dim, dim_head * heads)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        # self.to_k = nn.Linear(dim, kv_dim, bias = False)
        self.to_k = ComplexLinear(dim, kv_dim)
        self.proj_k = nn.Parameter(init_(torch.zeros([seq_len, k],dtype=torch.complex64)))

        self.share_kv = share_kv
        if not share_kv:
            # self.to_v = nn.Linear(dim, kv_dim, bias = False)
            self.to_v = ComplexLinear(dim, kv_dim)
            self.proj_v = nn.Parameter(init_(torch.zeros([seq_len, k],dtype=torch.complex64)))

        self.dropout = ComplexDropout(dropout)

        # self.to_out = nn.Linear(dim_head * heads, dim)
        self.to_out = ComplexLinear(dim_head * heads, dim)
        self.attention_block = complex_attn(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
    def forward(self, x, context = None, **kwargs):
        x_A = x.real
        x_B = x.imag

        x_aaa = self.attention_block(x_A, x_A, x_A)
        x_aab = self.attention_block(x_A, x_A, x_B)
        x_aba = self.attention_block(x_A, x_B, x_A)
        x_baa = self.attention_block(x_B, x_A, x_A)
        x_abb = self.attention_block(x_A, x_B, x_B)
        x_bab = self.attention_block(x_B, x_A, x_B)
        x_bba = self.attention_block(x_B, x_B, x_A)
        x_bbb = self.attention_block(x_B, x_B, x_B)

        x_A = x_aaa - x_abb - x_bab - x_bba
        x_B = -x_bbb + x_baa + x_aba + x_aab

        out = torch.complex(x_A,x_B)

        return out

class Linformer(nn.Module):
    def __init__(self, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.2):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = LinformerSelfAttention(dim, seq_len, k = k, heads = heads, dim_head = dim_head, one_kv_head = one_kv_head, share_kv = share_kv, dropout = dropout)
            ff = FeedForward(dim, dropout = dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)

class LinformerLM(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, k = 256, heads = 8, dim_head = None, one_kv_head = False, share_kv = False, reversible = False, dropout = 0.):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.linformer = Linformer(dim, seq_len, depth, k = k, heads = heads, dim_head = dim_head,
                one_kv_head = one_kv_head, share_kv = share_kv, reversible = reversible, dropout = dropout)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(x.shape[1], device=x.device)) + x
        x = self.linformer(x)
        out = self.to_logits(x)
        return out

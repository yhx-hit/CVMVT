import torch
from torch import nn, einsum
import torch.nn.functional as F
from .complexLayers import ComplexDropout
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .efficient import ComplexSequential, ComplexMaxPool2d,ComplexFlatten, ComplexLinear
from .complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU
# helpers
import math
from .linform import ComplexLayerNorm2d
from .complexLayers import ComplexReLU
from .efficient import ComplexLayerNorm1d

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        normfun = ComplexLayerNorm2d(x.shape[-1], elementwise_affine=False)
        x = normfun(x)
        return self.fn(x, **kwargs)

# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net = ComplexSequential(
            ComplexLinear(dim, hidden_dim),
            ComplexReLU(),
            ComplexDropout(dropout),
            ComplexLinear(hidden_dim, dim),
            ComplexDropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# attention

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, xq,xk,xv, context = None, kv_include_self = False):
        b, n, _, h = *xq.shape, self.heads
        context = default(context, xq)

        if kv_include_self:
            context = torch.cat((xq, context), dim = 1) # cross attention requires CLS token includes itself as key / value

        # qkv = (self.to_q(xq), *self.to_kv(context).chunk(2, dim = -1))
        qkv = (self.to_q(xq), self.to_k(xk), self.to_v(xv))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Attention0(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.attention_block = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)

    def forward(self, x, context = None, kv_include_self = False):
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


# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = ComplexLayerNorm2d(dim,elementwise_affine=False)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention0(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = ComplexLinear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = ComplexLinear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x

# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim, PreNorm(lg_dim, Attention0(lg_dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                ProjectInOut(lg_dim, sm_dim, PreNorm(sm_dim, Attention0(sm_dim, heads = heads, dim_head = dim_head, dropout = dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim = 1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim = 1)
        return sm_tokens, lg_tokens

# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth,
        sm_dim,
        lg_dim,
        sm_enc_params,
        lg_enc_params,
        cross_attn_heads,
        cross_attn_depth,
        cross_attn_dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim = sm_dim, dropout = dropout, **sm_enc_params),
                Transformer(dim = lg_dim, dropout = dropout, **lg_enc_params),
                # CrossTransformer(sm_dim = sm_dim, lg_dim = lg_dim, depth = cross_attn_depth, heads = cross_attn_heads, dim_head = cross_attn_dim_head, dropout = dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        # for sm_enc, lg_enc, cross_attend in self.layers:
        for sm_enc, lg_enc in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            # sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens

# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.h = image_size//patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
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
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(init_(torch.zeros([1, num_patches + 1, dim], dtype=torch.complex64)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(dropout)
        self.dropout = ComplexDropout(dropout)
        self.rang1 = Rearrange('b c (h p1) (w p2) -> (b h w c) p1 p2', p1=patch_size, p2=patch_size)
        self.linear1 = ComplexLinear(1152 * 3, dim)


    def forward(self, img):
        x = self.rang1(img)
        batch_size, n_features, n_features = x.shape
        x = x.reshape(-1, 1, n_features, n_features)
        x = self.conv(x)
        batchsize = img.shape[0]
        rang1 = Rearrange('(b h w c) p -> b (h w) (p c)', b=batchsize, c=3, h=self.h, w=self.h)
        x = rang1(x)
        x = self.linear1(x)
        # x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)
        cls_token = torch.complex(cls_tokens,cls_tokens2)
        x = torch.cat((cls_token, x), dim=1)
        # pos_embedding = torch.complex(self.pos_embedding,self.pos_embedding2)
        x += self.pos_embedding[:, :(n + 1)]

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class

class ImageEmbedder2(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        patch_size,
        dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        self.h = image_size//patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )
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
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(init_(torch.zeros([1, num_patches + 1, dim], dtype=torch.complex64)))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        # self.dropout = nn.Dropout(dropout)
        self.dropout = ComplexDropout(dropout)
        self.rang1 = Rearrange('b c (h p1) (w p2) -> (b h w c) p1 p2', p1=patch_size, p2=patch_size)
        self.linear1 = ComplexLinear(13824, dim)


    def forward(self, img):
        x = self.rang1(img)
        batch_size, n_features, n_features = x.shape
        x = x.reshape(-1, 1, n_features, n_features)
        x = self.conv(x)
        batchsize = img.shape[0]
        rang1 = Rearrange('(b h w c) p -> b (h w) (p c)', b=batchsize, c=3, h=self.h, w=self.h)
        x = rang1(x)
        x = self.linear1(x)
        # x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)
        cls_token = torch.complex(cls_tokens,cls_tokens2)
        x = torch.cat((cls_token, x), dim=1)
        # pos_embedding = torch.complex(self.pos_embedding,self.pos_embedding2)
        x += self.pos_embedding[:, :(n + 1)]

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)

# cross ViT class


class CrossViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        num_classes,
        sm_dim,
        lg_dim,
        sm_patch_size = 12,
        sm_enc_depth = 1,
        sm_enc_heads = 8,
        sm_enc_mlp_dim = 2048,
        sm_enc_dim_head = 64,
        lg_patch_size = 16,
        lg_enc_depth = 4,
        lg_enc_heads = 8,
        lg_enc_mlp_dim = 2048,
        lg_enc_dim_head = 64,
        cross_attn_depth = 2,
        cross_attn_heads = 8,
        cross_attn_dim_head = 64,
        depth = 3,
        dropout = 0.1,
        emb_dropout = 0.1
    ):
        super().__init__()
        self.sm_image_embedder = ImageEmbedder(dim = sm_dim, image_size = image_size, patch_size = sm_patch_size, dropout = emb_dropout)
        self.lg_image_embedder = ImageEmbedder2(dim = lg_dim, image_size = image_size, patch_size = lg_patch_size, dropout = emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth = depth,
            sm_dim = sm_dim,
            lg_dim = lg_dim,
            cross_attn_heads = cross_attn_heads,
            cross_attn_dim_head = cross_attn_dim_head,
            cross_attn_depth = cross_attn_depth,
            sm_enc_params = dict(
                depth = sm_enc_depth,
                heads = sm_enc_heads,
                mlp_dim = sm_enc_mlp_dim,
                dim_head = sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = lg_enc_depth,
                heads = lg_enc_heads,
                mlp_dim = lg_enc_mlp_dim,
                dim_head = lg_enc_dim_head
            ),
            dropout = dropout
        )

        # self.sm_mlp_head = nn.Sequential(nn.LayerNorm(sm_dim), nn.Linear(sm_dim, num_classes))
        # self.lg_mlp_head = nn.Sequential(nn.LayerNorm(lg_dim), nn.Linear(lg_dim, num_classes))
        # self.sm_mlp_head = ComplexSequential(ComplexLayerNorm1d(sm_dim,elementwise_affine=False))
        # self.lg_mlp_head = ComplexSequential(ComplexLayerNorm1d(lg_dim,elementwise_affine=False))
        # self.linear1 = ComplexLinear(sm_dim+lg_dim, num_classes)

        # self.sm_mlp_head = ComplexSequential(ComplexLayerNorm1d(sm_dim, elementwise_affine=False),ComplexLinear(sm_dim , num_classes))
        # self.lg_mlp_head = ComplexSequential(ComplexLayerNorm1d(lg_dim, elementwise_affine=False),ComplexLinear(lg_dim , num_classes))
        self.sm_mlp_head = ComplexSequential(ComplexLayerNorm1d(sm_dim, elementwise_affine=False))
        self.lg_mlp_head = ComplexSequential(ComplexLayerNorm1d(lg_dim, elementwise_affine=False))

        self.linear1 = ComplexLinear(sm_dim + lg_dim, num_classes)

    def forward(self, img):

        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)
        x = torch.cat((sm_logits,lg_logits),dim=1)
        return self.linear1(x)
        # return sm_logits+lg_logits

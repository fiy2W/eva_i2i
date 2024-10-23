import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./src/')
from eva.model.hyperconv2 import hyperConv


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class hyperBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, style_dim, latent_dim=1, layer_scale_init_value=1e-6):
        super().__init__()
        #self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.pad = nn.ReflectionPad2d(3)
        self.dwconv = hyperConv(style_dim, dim, dim, ksize=7, padding=0, groups=dim, weight_dim=latent_dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = hyperConv(style_dim, dim, dim*4, ksize=1, padding=0, weight_dim=latent_dim)
        self.act = nn.GELU()
        self.pwconv2 = hyperConv(style_dim, dim*4, dim, ksize=1, padding=0, weight_dim=latent_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1,dim,1,1)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, s):
        input = x
        x = self.pad(x)
        x = self.dwconv(x, s)
        #x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, s)
        x = self.act(x)
        x = self.pwconv2(x, s)
        
        if self.gamma is not None:
            x = self.gamma * x

        x = input + x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0, bias=False)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1, bias=False)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1, bias=False)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class AttnResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, n_layer, layer_scale_init_value=1e-6):
        super().__init__()

        self.n_layer = n_layer
        
        if n_layer>0:
            self.block1 = Block(dim, layer_scale_init_value=layer_scale_init_value)
        else:
            self.block1 = nn.Identity()
        
        self.attnblocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for _ in range(n_layer - 1):
            self.attnblocks.append(AttnBlock(dim))
            self.resblocks.append(Block(dim, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x):
        x = self.block1(x)

        for attn, res in zip(self.attnblocks, self.resblocks):
            x = attn(x)
            x = res(x)
        return x
    

class hyperAttnResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, style_dim, n_layer, latent_dim=1, layer_scale_init_value=1e-6):
        super().__init__()

        self.n_layer = n_layer
        
        if n_layer>0:
            self.block1 = hyperBlock(dim, style_dim, latent_dim=latent_dim, layer_scale_init_value=layer_scale_init_value)
        else:
            self.block1 = nn.Identity()
        
        self.attnblocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for _ in range(n_layer - 1):
            self.attnblocks.append(AttnBlock(dim))
            self.resblocks.append(hyperBlock(dim, style_dim, latent_dim=latent_dim, layer_scale_init_value=layer_scale_init_value))

    def forward(self, x, s):
        if self.n_layer==0:
            x = self.block1(x)
        else:
            x = self.block1(x, s)

        for attn, res in zip(self.attnblocks, self.resblocks):
            x = attn(x)
            x = res(x, s)
        return x
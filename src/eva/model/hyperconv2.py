import torch
import torch.nn as nn
import torch.nn.functional as F


class hyperConv(nn.Module):
    def __init__(
        self,
        style_dim,
        dim_in,
        dim_out,
        ksize,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        ndims=2,
        latent_dim=1,
        eps=1e-9,
    ):
        super().__init__()
        assert ndims in [2, 3]
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.latent_dim = latent_dim
        self.fc_k = nn.Linear(style_dim, ksize*ksize) if self.ndims==2 else nn.Linear(style_dim, ksize*ksize*ksize)
        self.fc_cout = nn.Linear(style_dim, dim_out*latent_dim)
        self.fc_cin = nn.Linear(style_dim, dim_in//groups*latent_dim)
        self.kshape = [dim_out, dim_in//groups, ksize, ksize] if self.ndims==2 else [dim_out, dim_in//groups, ksize, ksize, ksize]
        self.padding = (ksize-1)//2 if padding is None else padding
        self.groups = groups
        self.dilation = dilation
        self.eps = eps

        self.weight = nn.Parameter(torch.randn(self.kshape).type(torch.float32))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in')

        self.conv = getattr(F, 'conv%dd' % self.ndims)
            
    def forward(self, x, s):
        bs = x.shape[0]
        #dwh = x.shape[2:]

        f_k = self.fc_k(s).view(bs, *self.kshape[2:])
        w_k = f_k[:, None, None, :, :] if self.ndims==2 else f_k[:, None, None, :, :, :]

        f_cout = self.fc_cout(s).view(bs, self.dim_out, self.latent_dim)
        f_cin = self.fc_cin(s).view(bs, self.latent_dim, -1)
        f_cout_cin = f_cout @ f_cin
        w_cout_cin = f_cout_cin[:, :, :, None, None] if self.ndims==2 else f_cout[:, :, :, None, None, None]

        w1 = self.weight[None, :, :, :, :] if self.ndims==2 else self.weight[None, :, :, :, :, :]
        weights = w1 * w_cout_cin * w_k
        #d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps) if self.ndims==2 else torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
        #weights = weights * d

        out = []
        for i in range(bs):
            out.append(self.conv(x[i:i+1], weight=weights[i], stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups))
        x = torch.cat(out, dim=0)
        #x = x.reshape(1, -1, *dwh)
        #_, _, *ws = weights.shape
        #weights = weights.reshape(bs * self.dim_out, *ws)
        #x = self.conv(x, weight=weights, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups*bs)
        #dwh = x.shape[2:]
        #x = x.reshape(bs, self.dim_out, *dwh)
        return x


class hyperConvTranspose(nn.Module):
    def __init__(
        self,
        style_dim,
        dim_in,
        dim_out,
        ksize,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        groups=1,
        ndims=2,
        eps=1e-9,
    ):
        super().__init__()
        assert ndims in [2, 3]
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.fc_cin = nn.Linear(style_dim, dim_in)
        self.fc_cout = nn.Linear(style_dim, dim_out//groups)
        self.fc_k = nn.Linear(style_dim, ksize*ksize) if self.ndims==2 else nn.Linear(style_dim, ksize*ksize*ksize)
        self.kshape = [dim_in, dim_out//groups, ksize, ksize] if self.ndims==2 else [dim_in, dim_out//groups, ksize, ksize, ksize]
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation

        self.param = nn.Parameter(torch.randn(self.kshape).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')

        self.conv = getattr(F, 'conv_transpose%dd' % self.ndims)
            
    def forward(self, x, s):
        if s.shape[0]==1:
            return self.forwart_bs1(x, s)
        elif s.shape[0]==x.shape[0]:
            out = []
            for i in range(s.shape[0]):
                out.append(self.forwart_bs1(x[i:i+1], s[i:i+1]))
            out = torch.cat(out, dim=0)
            return out
    
    def forwart_bs1(self, x, s):
        bs = x.shape[0]
        dwh = x.shape[2:]

        f_cin = self.fc_cin(s)
        f_cout = self.fc_cout(s)
        f_k = self.fc_k(s).view(-1, *self.kshape[2:])
        
        w_cout = f_cout[:, None, :, None, None] if self.ndims==2 else f_cout[:, None, :, None, None, None]
        w_cin = f_cin[:, :, None, None, None] if self.ndims==2 else f_cin[:, :, None, None, None, None]
        w_k = f_k[:, None, None, :, :] if self.ndims==2 else f_k[:, None, None, :, :, :]

        w1 = self.weight[None, :, :, :, :] if self.ndims==2 else self.weight[None, :, :, :, :, :]
        weights = w1 * (w_cin + 1) * (w_cout + 1) * (w_k + 1)
        d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps) if self.ndims==2 else torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4, 5), keepdim=True) + self.eps)
        weights = weights * d

        x = x.reshape(1, -1, *dwh)
        _, _, *ws = weights.shape
        weights = weights.reshape(bs * self.dim_out, *ws)
        x = self.conv(x, weight=weights, stride=self.stride, padding=self.padding, output_padding=self.output_padding, dilation=self.dilation, groups=self.groups*bs)
        dwh = x.shape[2:]
        x = x.reshape(bs, self.dim_out, *dwh)

        return x
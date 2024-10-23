import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('./src/')
from eva.model.convnext import Block, LayerNorm, hyperAttnResBlock, AttnResBlock


class ImageEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.c_in = args['c_in']
        self.c_enc = args['c_enc']
        self.k_enc = args['k_enc']
        self.s_enc = args['s_enc']
        self.b_enc = args['b_enc']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.embed_dim = args['embed_dim']
        self.heads = args['transformer_heads']

        self.layers = nn.ModuleList()
        c_pre = self.c_in
        for i, (c, k, s, b) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.b_enc)):
            if i==0:
                block = [
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
                if b!=0:
                    block.append(LayerNorm(c, eps=1e-6, data_format="channels_first"))
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
            block.extend([Block(dim=c, layer_scale_init_value=self.layer_scale_init_value) for _ in range(b)])
            self.layers.append(nn.Sequential(*block))
            c_pre = c
        
        self.norm = LayerNorm(c, eps=1e-6, data_format="channels_first")
        self.attnpool = AttentionPool2d(8, c, self.heads, self.embed_dim)

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            
        x = self.norm(x)
        #x = F.interpolate(x, size=7, mode='bilinear', align_corners=True)
        x = self.attnpool(x)
        x = x / x.norm(dim=1, keepdim=True)
        return x, features


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class AttnEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c_in = args['c_in']
        self.c_enc = args['c_enc']
        self.k_enc = args['k_enc']
        self.s_enc = args['s_enc']
        self.b_enc = args['b_enc']
        self.layer_scale_init_value = args['layer_scale_init_value']

        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        c_pre = self.c_in
        for i, (c, k, s, b) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.b_enc)):
            if i==0:
                block = [
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
                if b!=0:
                    block.append(LayerNorm(c, eps=1e-6, data_format="channels_first"))
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
            self.layers_1.append(nn.Sequential(*block))
            self.layers_2.append(AttnResBlock(c, b, layer_scale_init_value=self.layer_scale_init_value))
            c_pre = c

    def forward(self, x):
        for layer1, layer2 in zip(self.layers_1, self.layers_2):
            x = layer1(x)
            x = layer2(x)

        return x
    

class hyperEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.style_dim = args['embed_dim']
        
        self.c_in = args['c_in']
        self.c_enc = args['c_enc']
        self.k_enc = args['k_enc']
        self.s_enc = args['s_enc']
        self.b_enc = args['b_enc']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.embed_dim = args['embed_dim']
        self.latent_dim = args['latent_dim']

        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        c_pre = self.c_in
        for i, (c, k, s, b) in enumerate(zip(self.c_enc, self.k_enc, self.s_enc, self.b_enc)):
            if i==0:
                block = [
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
                if b!=0:
                    block.append(LayerNorm(c, eps=1e-6, data_format="channels_first"))
            else:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
            self.layers_1.append(nn.Sequential(*block))
            self.layers_2.append(hyperAttnResBlock(c, self.embed_dim, b, latent_dim=self.latent_dim, layer_scale_init_value=self.layer_scale_init_value))
            c_pre = c

    def forward(self, x, s):
        for layer1, layer2 in zip(self.layers_1, self.layers_2):
            x = layer1(x)
            x = layer2(x, s)

        return x
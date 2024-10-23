import torch
import torch.nn as nn

import sys
sys.path.append('./src/')
from eva.model.convnext import LayerNorm, hyperAttnResBlock


class hyperDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.embed_dim = args['embed_dim']

        self.c_dec = args['c_dec']
        self.k_dec = args['k_dec']
        self.s_dec = args['s_dec']
        self.b_dec = args['b_dec']
        self.layer_scale_init_value = args['layer_scale_init_value']
        self.latent_dim = args['latent_dim']

        self.layers_1 = nn.ModuleList()
        self.layers_2 = nn.ModuleList()
        c_pre = args['c_pre']

        for c, k, s, b in zip(self.c_dec, self.k_dec, self.s_dec, self.b_dec):
            self.layers_1.append(hyperAttnResBlock(c_pre, self.embed_dim, b, latent_dim=self.latent_dim, layer_scale_init_value=self.layer_scale_init_value))
            if s==1:
                block = [
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=s, padding_mode='reflect'),
                ]
                self.layers_2.append(nn.Sequential(*block))
            else:
                self.layers_2.append(nn.Sequential(*[
                    LayerNorm(c_pre, eps=1e-6, data_format="channels_first"),
                    nn.Upsample(scale_factor=s, mode='nearest'),
                    nn.Conv2d(in_channels=c_pre, out_channels=c, kernel_size=k, padding=(k-1)//2, stride=1, padding_mode='reflect'),
                ]))
            
            c_pre = c

        if args['act_last']=='None':
            self.act_last = nn.Identity()
        elif args['act_last']=='tanh':
            self.act_last = nn.Tanh()
        elif args['act_last']=='sigmoid':
            self.act_last = nn.Sigmoid()
        elif args['act_last']=='relu':
            self.act_last = nn.ReLU()
        elif args['act_last']=='gelu':
            self.act_last = nn.GELU()
        elif args['act_last']=='leakyrelu':
            self.act_last = nn.LeakyReLU(0.01)

    def forward(self, x, s):
        for layer1, layer2 in zip(self.layers_1, self.layers_2):
            x = layer1(x, s)
            x = layer2(x)
        
        x = self.act_last(x)
        return x
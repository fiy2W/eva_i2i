import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.layers import trunc_normal_

import sys
sys.path.append('./src/')
from eva.model.image_encoder import ImageEncoder, AttnEncoder, hyperAttnResBlock, LayerNorm
from eva.model.image_decoder import hyperDecoder
from eva.model.text_encoder import TextEncoder, _tokenizer


class EVA(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.prompt_encoder = TextEncoder(
            args['text_encoder']['embed_dim'],
            context_length=args['text_encoder']['context_length'],
            vocab_size=len(_tokenizer.encoder),
            transformer_heads=args['text_encoder']['transformer_heads'],
            transformer_layers=args['text_encoder']['transformer_layers'],
            transformer_width=args['text_encoder']['transformer_width'],
        )
        self.image_encoder = ImageEncoder(args['style_encoder'])
        self.seq2seq = Seq2Seq(args)

        self.act = nn.LeakyReLU(0.01)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x_src, x_tgt, s_src, s_tgt, recon=False):
        if recon:
            with torch.no_grad():
                style_src = self.encode_text(s_src).detach()
                style_tgt = self.encode_text(s_tgt).detach()
            
            x_src = x_src * 2. - 1.
            x_tgt = x_tgt * 2. - 1.
            rec_src2tgt, c_src = self.seq2seq(x_src, style_tgt)
            rec_tgt2src, c_tgt = self.seq2seq(x_tgt, style_src)
            rec_src2tgt = self.act(rec_src2tgt * 0.5 + 0.5)
            rec_tgt2src = self.act(rec_tgt2src * 0.5 + 0.5)
            return rec_src2tgt, rec_tgt2src, c_src, c_tgt, style_src, style_tgt
        else:
            image_features = self.encode_image(x_src)
            text_features = self.encode_text(s_src)
            return image_features, text_features
    
    def infer_triangle(self, x_src, s_tgt1, s_tgt2):
        with torch.no_grad():
            style_tgt1 = self.encode_text(s_tgt1).detach()
            style_tgt2 = self.encode_text(s_tgt2).detach()
        
        x_src = x_src * 2. - 1.
        rec_src2tgt1, c_src = self.seq2seq(x_src, style_tgt1)
        rec_tgt2tgt2, c_tgt1 = self.seq2seq(torch.clamp(rec_src2tgt1, min=-1), style_tgt2)
        rec_src2tgt1 = self.act(rec_src2tgt1 * 0.5 + 0.5)
        rec_tgt2tgt2 = self.act(rec_tgt2tgt2 * 0.5 + 0.5)
        return rec_src2tgt1, rec_tgt2tgt2, c_src, c_tgt1, style_tgt1, style_tgt2
    
    @torch.no_grad()
    def infer_syn(self, x, s_tgt):
        style_tgt = self.encode_text(s_tgt)
        
        x = x * 2. - 1.
        rec, _ = self.seq2seq(x, style_tgt)
        rec = self.act(rec * 0.5 + 0.5)
        return rec, style_tgt

    def encode_image(self, x):
        x = x * 2. - 1.
        image_features, _ = self.image_encoder(x)
        return image_features
    
    def encode_text(self, s):
        text_features, _ = self.prompt_encoder(s)
        return text_features


class Seq2Seq(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.hyper_encoder = AttnEncoder(args['content_encoder'])
        self.hyper_decoder = hyperDecoder(args['recon_decoder'])
    
    def forward(self, x_src, style_tgt):
        c_src = self.hyper_encoder(x_src)
        rec_src2tgt = self.hyper_decoder(c_src, style_tgt)
        return rec_src2tgt, c_src


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, args):
        super(NLayerDiscriminator, self).__init__()

        style_dim = args['embed_dim']
        layer_scale_init_value = args['layer_scale_init_value']
        c_in = args['c_in']
        ndf = args['ndf']
        n_layers = args['n_down']
        n_hyper = args['n_hyper']
        latent_dim = args['latent_dim']

        kw = 4
        padw = 1

        sequence = [nn.Conv2d(c_in, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers-1):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                LayerNorm(ndf * nf_mult, eps=1e-6, data_format="channels_first"),
                nn.LeakyReLU(0.2, True),
            ]

        self.encoder = nn.Sequential(*sequence)
        self.hyperblocks = hyperAttnResBlock(ndf * nf_mult, style_dim, n_hyper, latent_dim=latent_dim, layer_scale_init_value=layer_scale_init_value)
        self.conv_out = nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # output 1 channel prediction map
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, s):
        x = self.encoder(input)
        x = self.hyperblocks(x, s)
        x = self.conv_out(x)

        return x
    

def content_contrastive_loss(train_imgs, train_target_imgs, p_src, p_tgt):
    contrastive_loss = SupConLoss()
    mask1 = (((train_imgs<=0)+(train_target_imgs<=0))<0.5).to(device=train_imgs.device, dtype=torch.float32).reshape(-1, *train_imgs.shape[1:])
    mask1 = F.interpolate(mask1, scale_factor=0.25, mode='nearest')>0.5
    #pos = torch.nonzero(mask1)
    p_src = p_src#.mode()
    p_tgt = p_tgt#.mode()
    bs, c = p_src.shape[0:2]
    p_src = p_src / p_src.norm(dim=1, keepdim=True)
    p_tgt = p_tgt / p_tgt.norm(dim=1, keepdim=True)

    p_src = p_src.permute(0,2,3,1).reshape(bs, -1, c)
    p_tgt = p_tgt.permute(0,2,3,1).reshape(bs, -1, c)
    mask1 = mask1.permute(0,2,3,1).reshape(bs, -1,)

    loss = 0
    n = 1024
    for i in range(bs):
        p_src_1 = p_src[i][mask1[i]]
        p_tgt_1 = p_tgt[i][mask1[i]]
        if p_src_1.shape[0]>=2:
            loss += contrastive_loss(torch.stack([p_src_1[:n], p_tgt_1[:n]], dim=1))
    return loss/bs

from typing import List, Optional, Tuple, Union
from skimage.util import random_noise

import torch
from torch import Tensor


def _blend(img1: Tensor, img2: Tensor, ratio: Tensor) -> Tensor:
    return (ratio * img1 + (1.0 - ratio) * img2).to(img1.dtype)


def adjust_brightness(img: Tensor, brightness_factor: Tensor) -> Tensor:
    brightness_factor = brightness_factor.to(device=img.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return _blend(img, torch.zeros_like(img), brightness_factor)


def adjust_contrast(img: Tensor, contrast_factor: Tensor) -> Tensor:
    contrast_factor = contrast_factor.to(device=img.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    mean = torch.mean(img, dim=(1,2,3), keepdim=True)
    return _blend(img, mean, contrast_factor)


def adjust_gamma(img: Tensor, gamma: Tensor) -> Tensor:
    gamma = gamma.to(device=img.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    img = (img**gamma)
    return img


def add_gauss_noise(img: Tensor, mean: float, std: float) -> Tensor:
    noise = torch.normal(mean, std, size=img.shape).to(device=img.device)
    return img + noise


def random_aug(imgs: Tensor) -> Tensor:
    mask_brightness = torch.rand(imgs.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=imgs.device)>0.5
    mask_contrast = torch.rand(imgs.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=imgs.device)>0.5
    mask_gamma = torch.rand(imgs.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=imgs.device)>0.5
    mask_gauss = torch.rand(imgs.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=imgs.device)>0.5
    mask_aug = torch.rand(imgs.shape[0]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device=imgs.device)>0.5

    imgs_bright = adjust_brightness(imgs, brightness_factor=1+(torch.rand(imgs.shape[0])+0.5)/50)
    imgs_aug = mask_brightness*imgs_bright + (~mask_brightness)*imgs

    imgs_contrast = adjust_contrast(imgs_aug, contrast_factor=1+(torch.rand(imgs.shape[0])+0.5)/50)
    imgs_aug = mask_contrast*imgs_contrast + (~mask_contrast)*imgs_aug

    imgs_gamma = adjust_gamma(torch.clamp(imgs_aug, min=0), gamma=1+(torch.rand(imgs.shape[0])+0.5)/50)
    imgs_aug = mask_gamma*imgs_gamma + (~mask_gamma)*imgs_aug

    imgs_gauss = add_gauss_noise(imgs_aug, mean=0, std=0.05)
    imgs_aug = mask_gauss*imgs_gauss + (~mask_gauss)*imgs_aug

    imgs_aug = mask_aug*imgs_aug + (~mask_aug)*imgs
    return imgs_aug
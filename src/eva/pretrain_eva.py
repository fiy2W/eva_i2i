import argparse
import logging
import numpy as np
import os
import yaml
import math
import random
from tqdm import tqdm
from collections import Counter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision

import sys
sys.path.append('./src/')
from eva.model.eva import EVA, NLayerDiscriminator, content_contrastive_loss
from eva.model.ema import EMAModel
from eva.model.text_encoder import tokenize
from eva.dataset.dataloader import Dataset_ALL
from eva.dataset.augmentation import random_aug
from eva.utils import Recorder, Plotter, load_weights, torch_PSNR
from eva.losses import SupConLoss, PerceptualLoss, GANLoss


def train(args_data, args, net, netd, device):
    # fix the seed for reproducibility
    seed = args['pretrain_eva']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = Dataset_ALL(args_data, mode='train', test_fold=args['test_fold'], valid_fold=args['valid_fold'])
    
    n_train = len(train_data)
    label_counts = Counter(train_data.dataset_label)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    weights = [class_weights[train_data.dataset_label[i]] for i in range(n_train)]
    weights = torch.DoubleTensor(weights)
    
    train_loader = DataLoader(train_data, batch_size=args['pretrain_eva']['batch_size'], num_workers=args['pretrain_eva']['num_workers'], pin_memory=True,
        sampler=WeightedRandomSampler(weights, n_train), collate_fn=lambda x: x)
    
    valid_data = Dataset_ALL(args_data, mode='valid', test_fold=args['test_fold'], valid_fold=args['valid_fold'])
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=args['pretrain_eva']['num_workers'], pin_memory=True, collate_fn=lambda x: x)

    
    n_valid = len(valid_data)

    size = args_data['prep']['size']
    batch_size = args['pretrain_eva']['batch_size']
    epochs = args['pretrain_eva']['epochs']
    lr = np.float32(args['pretrain_eva']['lr'])
    dir_visualize = args['pretrain_eva']['vis']
    dir_checkpoint = args['pretrain_eva']['ckpt']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Valid size:      {n_valid}
        Device:          {device.type}
    ''')

    perceptual = PerceptualLoss().to(device=device)
    gan = GANLoss('lsgan').to(device=device)
    n_step_pre_epoch = n_train//batch_size

    optimizer = torch.optim.AdamW(net.seq2seq.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args['pretrain_eva']['weight_decay'])
    optimizerD = torch.optim.AdamW(netd.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args['pretrain_eva']['weight_decay'])
    model_ema = EMAModel(net.parameters())
    model_ema.to(device=device)

    recorder = Recorder(['rec', 'per', 'content', 'psnr', 'dis', 'adv'])
    plotter = Plotter(dir_visualize, keys2=['psnr'], keys1=['per', 'rec', 'content', 'dis', 'adv'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,rec,per,content,psnr\n')

    total_step = 0
    best_loss = 0
    for epoch in range(epochs):
        net.train()
        netd.train()
        train_metrics = {
            'total': [], 'rec': [], 'per': [], 'content': [], 'dis': [], 'adv': [],
        }
        with tqdm(total=n_step_pre_epoch*args['pretrain_eva']['repeat_sample'], desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                data_unit = {}

                #batch = next(loader)
                for batch_1 in batch:
                    box_b = batch_1['box']
                    organ = batch_1['organ']
                    flags = batch_1['flag']
                    flags_random = deepcopy(flags)
                    if len(flags_random)==2:
                        flags_random = flags_random[::-1]
                    else:
                        random.shuffle(flags_random)

                    if organ not in data_unit:
                        data_unit[organ] = []
                    for batch_i, batch_ri in zip(flags, flags_random):
                        batch_i = int(batch_i)
                        batch_ri = int(batch_ri)
                        data_unit[organ].append({
                            'src_imgs': batch_1[batch_i].unsqueeze(0),
                            'src_prompts': batch_1['prompt'][batch_i],
                            'tgt_imgs': batch_1[batch_ri].unsqueeze(0),
                            'tgt_prompts': batch_1['prompt'][batch_ri],
                            'boxes': box_b,
                            'dataset': batch_1['dataset'],
                        })

                for _ in range(args['pretrain_eva']['repeat_sample']):
                    train_data_unit = []
                    
                    plane_dict = ['axial', 'coronal', 'sagittal']
                    for plane in range(3):
                        view_prompt = ' The image is visualized in the {} plane.'.format(plane_dict[plane])
                        for _ in range(args['pretrain_eva']['repeat_slice']):
                            for organ in data_unit.keys():
                                for unit in data_unit[organ]:
                                    src_img = unit['src_imgs']
                                    src_prom = unit['src_prompts'].copy()
                                    tgt_img = unit['tgt_imgs']
                                    tgt_prom = unit['tgt_prompts'].copy()
                                    box = unit['boxes']

                                    int_unit = random.choice(data_unit[organ])
                                    int_img = int_unit['src_imgs']
                                    int_prom = int_unit['src_prompts'].copy()

                                    sid = random.randint(box[0+plane*2], box[1+plane*2])
                                    sid = max(min(sid, size[plane]-1), 0)
                                    if plane==0:
                                        src_img_slice = src_img[:,:,sid]
                                        tgt_img_slice = tgt_img[:,:,sid]
                                        int_img_slice = int_img[:,:,sid]
                                    elif plane==1:
                                        src_img_slice = src_img[:,:,:,sid]
                                        tgt_img_slice = tgt_img[:,:,:,sid]
                                        int_img_slice = int_img[:,:,:,sid]
                                    elif plane==2:
                                        src_img_slice = src_img[:,:,:,:,sid]
                                        tgt_img_slice = tgt_img[:,:,:,:,sid]
                                        int_img_slice = int_img[:,:,:,:,sid]
                                    
                                    src_newprom = src_prom + [{'content': [view_prompt], 'mask': 1}]
                                    tgt_newprom = tgt_prom + [{'content': [view_prompt], 'mask': 1}]
                                    int_newprom = int_prom + [{'content': [view_prompt], 'mask': 1}]
                                    atlas_newprom = [{'content': ['A T1-weighted magnetic resonance imaging of the {} for the subject.'.format(organ)], 'mask': 1}, {'content': [view_prompt], 'mask': 1}]

                                    train_data_unit.append({
                                        'src_imgs': src_img_slice,
                                        'src_prompts': src_newprom,
                                        'src_text': train_data.gen_text_from_prompts(src_newprom),
                                        'tgt_imgs': tgt_img_slice,
                                        'tgt_prompts': tgt_newprom,
                                        'tgt_text': train_data.gen_text_from_prompts(tgt_newprom),
                                        'int_imgs': int_img_slice,
                                        'int_prompts': int_newprom,
                                        'int_text': train_data.gen_text_from_prompts(int_newprom),
                                        'atlas_text': train_data.gen_text_from_prompts(atlas_newprom),
                                    })

                    random.shuffle(train_data_unit)

                    N = min(args['pretrain_eva']['dec_num'], len(train_data_unit))
                    train_src_imgs = [train_data_unit[i]['src_imgs'] for i in range(N)]
                    train_src_prompts = [train_data_unit[i]['src_text'] for i in range(N)]
                    train_tgt_imgs = [train_data_unit[i]['tgt_imgs'] for i in range(N)]
                    train_tgt_prompts = [train_data_unit[i]['tgt_text'] for i in range(N)]
                    train_int_imgs = [train_data_unit[i]['int_imgs'] for i in range(N)]
                    train_int_prompts = [train_data_unit[i]['int_text'] for i in range(N)]
                    train_atlas_prompts = [train_data_unit[i]['atlas_text'] for i in range(N)]

                    train_src_imgs = torch.cat(train_src_imgs, dim=0).to(device=device, dtype=torch.float32)
                    train_tgt_imgs = torch.cat(train_tgt_imgs, dim=0).to(device=device, dtype=torch.float32)
                    train_int_imgs = torch.cat(train_int_imgs, dim=0).to(device=device, dtype=torch.float32)
                    train_src_tokens = tokenize(train_src_prompts, context_length=args['context_length']).to(device=device)
                    train_tgt_tokens = tokenize(train_tgt_prompts, context_length=args['context_length']).to(device=device)
                    train_int_tokens = tokenize(train_int_prompts, context_length=args['context_length']).to(device=device)
                    train_atlas_tokens = tokenize(train_atlas_prompts, context_length=args['context_length']).to(device=device)
                    #train_random_tokens = tokenize(train_random_prompts, context_length=args['context_length']).to(device=device)
                    #train_labels = torch.from_numpy(np.array(train_labels, dtype=np.int64)).to(device=device)

                    train_src_imgs_aug = random_aug(train_src_imgs)
                    train_tgt_imgs_aug = random_aug(train_tgt_imgs)
                    train_int_imgs_aug = random_aug(train_int_imgs)

                    mask = (((train_src_imgs>0)*(train_tgt_imgs<=0)+(train_src_imgs<=0)*(train_tgt_imgs>0))<0.5).to(device=device, dtype=torch.float32)
                    mask4 = F.interpolate(mask, scale_factor=0.25, mode='nearest')
                    mask1 = (((train_src_imgs<=0)+(train_tgt_imgs<=0))<0.5).to(device=train_src_imgs.device, dtype=torch.float32).reshape(-1, *train_src_imgs.shape[1:])

                    # train discriminator
                    if epoch>=args['start_dis']:
                        with torch.no_grad():
                            rec_src2tgt, rec_tgt2src, _, _, txt_emb_src, txt_emb_tgt = net(train_src_imgs_aug, train_tgt_imgs_aug, train_src_tokens, train_tgt_tokens, recon=True)
                            rec_src2int, rec_int2tgt, _, _, txt_emb_int, _ = net.infer_triangle(train_src_imgs_aug, train_int_tokens, train_tgt_tokens)
                            rec_src2atl, rec_atl2int, _, _, _, _ = net.infer_triangle(train_src_imgs_aug, train_atlas_tokens, train_int_tokens)
                        pred_real = netd(train_tgt_imgs_aug*mask, txt_emb_tgt)
                        pred_fake = netd(torch.clamp(rec_src2tgt, min=0).detach()*mask, txt_emb_tgt)
                        pred_real2 = netd(train_src_imgs_aug*mask, txt_emb_src)
                        pred_fake2 = netd(torch.clamp(rec_tgt2src, min=0).detach()*mask, txt_emb_src)
                        pred_real3 = netd(train_int_imgs_aug*mask, txt_emb_int)
                        pred_fake3 = netd(torch.clamp(rec_src2int, min=0).detach()*mask, txt_emb_int)
                        pred_fake4 = netd(torch.clamp(rec_int2tgt, min=0).detach()*mask, txt_emb_tgt)

                        loss_d = (gan(pred_real, True)*2 + gan(pred_fake, False) + gan(pred_real2, True) + gan(pred_fake2, False) + \
                                  gan(pred_real3, True) + gan(pred_fake3, False) + gan(pred_fake4, False)) * 0.5
                        optimizerD.zero_grad()
                        loss_d.backward()
                        optimizerD.step()

                    rec_src2tgt, rec_tgt2src, p_src, p_tgt, _, _ = net(train_src_imgs_aug, train_tgt_imgs_aug, train_src_tokens, train_tgt_tokens, recon=True)
                    rec_src2int, rec_int2tgt, _, p_int, _, _ = net.infer_triangle(train_src_imgs_aug, train_int_tokens, train_tgt_tokens)
                    int_img_f = net.encode_image(torch.clamp(rec_src2int, min=0))
                    
                    loss_rec = nn.L1Loss()(rec_src2tgt*mask, train_tgt_imgs*mask) + nn.L1Loss()(rec_tgt2src*mask, train_src_imgs*mask) + nn.L1Loss()(rec_int2tgt*mask, train_tgt_imgs*mask)
                    loss_per = perceptual(rec_src2tgt*mask, train_tgt_imgs*mask) + perceptual(rec_tgt2src*mask, train_src_imgs*mask) + perceptual(rec_int2tgt*mask, train_tgt_imgs*mask)
                    loss_content = 10*nn.MSELoss()(p_src*mask4, p_tgt.detach()*mask4) + 10*nn.MSELoss()(p_int*mask4, p_tgt.detach()*mask4) + \
                        10*nn.MSELoss()(p_src.detach()*mask4, p_tgt*mask4) + 10*nn.MSELoss()(p_int.detach()*mask4, p_tgt*mask4) + \
                        content_contrastive_loss(train_src_imgs, train_tgt_imgs, p_src, p_tgt) + \
                        content_contrastive_loss(rec_int2tgt, train_tgt_imgs, p_int, p_tgt)

                    if epoch>=args['start_dis']:
                        pred_fake = netd(torch.clamp(rec_src2tgt, min=0)*mask, txt_emb_tgt)
                        pred_fake2 = netd(torch.clamp(rec_tgt2src, min=0)*mask, txt_emb_src)
                        pred_fake3 = netd(torch.clamp(rec_src2int, min=0), txt_emb_int)
                        pred_fake4 = netd(torch.clamp(rec_int2tgt, min=0)*mask, txt_emb_tgt)

                        loss_adv = gan(pred_fake, True) + gan(pred_fake2, True) + gan(pred_fake3, True) + gan(pred_fake4, True)
                        loss_clip =  - nn.CosineSimilarity()(int_img_f, txt_emb_int).mean()
                        loss_rec += nn.L1Loss()(rec_src2int*mask, torch.clamp(rec_atl2int, min=0).detach()*mask)
                    else:
                        loss_adv = loss_rec
                        loss_d = loss_rec
                        loss_clip = loss_rec
                        rec_src2atl = rec_src2int
                        rec_atl2int = rec_src2int
                    loss = args['pretrain_eva']['lambda_clip']*loss_clip + args['pretrain_eva']['lambda_rec']*loss_rec + args['pretrain_eva']['lambda_per']*loss_per + args['pretrain_eva']['lambda_content']*loss_content + args['pretrain_eva']['lambda_adv']*loss_adv

                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()))
                        sys.exit(1)

                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2)
                    optimizer.step()
                    model_ema.step(net.parameters())

                    train_metrics['total'].append(loss.item())
                    train_metrics['per'].append(loss_per.item())
                    train_metrics['rec'].append(loss_rec.item())
                    train_metrics['content'].append(loss_content.item())
                    train_metrics['dis'].append(loss_d.item())
                    train_metrics['adv'].append(loss_adv.item())
                    pbar.set_postfix(**{'clip': loss_clip.item(), 'dis': loss_d.item(), 'adv': loss_adv.item(), 'rec': loss_rec.item(), 'per': loss_per.item(), 'content': loss_content.item(), 'W': [round(i, 1) for i in perceptual.W_init]})
                    pbar.update(1)

                    if (total_step % args['pretrain_eva']['view_freq']) == 0:
                        with torch.no_grad():
                            vimage = torch.stack([
                                train_src_imgs, rec_src2tgt, train_tgt_imgs, rec_tgt2src, train_src_imgs, mask, mask, mask1,
                                train_src_imgs, rec_src2atl, rec_atl2int, rec_src2int, train_int_imgs, rec_int2tgt, train_tgt_imgs, mask1], dim=1).reshape(-1,1,size[0],size[1])
                            vimage = torch.clamp(vimage, min=0, max=1)
                            torchvision.utils.save_image(vimage, os.path.join(dir_visualize, '{}.jpg'.format(epoch)))
                    
                    if (total_step % args['pretrain_eva']['save_freq']) == 0:
                        model_ema.store(net.parameters())
                        model_ema.copy_to(net.parameters())
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))
                        model_ema.restore(net.parameters())

                        torch.save(netd.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp_d.pth'))

                    total_step += 1
                #break
                #torch.cuda.empty_cache()
        
        net.eval()
        model_ema.store(net.parameters())
        model_ema.copy_to(net.parameters())
        test_psnr = []
        with torch.no_grad():
            for batch in valid_loader:
                batch = batch[0]
                data_unit = []

                box_b = batch['box']
                if len(batch['flag'])==1:
                    continue
                for batch_i in batch['flag']:
                    batch_i = int(batch_i)
                    data_unit.append({
                        'imgs': batch[batch_i].unsqueeze(0),
                        'prompts': batch['prompt'][batch_i],
                        'boxes': box_b,
                        'dataset': batch['dataset'],
                        'organ': batch['organ'],
                    })

                plane_dict = ['axial', 'coronal', 'sagittal']
                for plane in range(3):
                    view_prompt = ' The image is visualized in the {} plane.'.format(plane_dict[plane])
                    valid_data_unit = []
                    for unit in data_unit:
                        img = unit['imgs']
                        prom = unit['prompts'].copy()
                        sid = img.shape[-1]//2
                        if plane==0:
                            img_slice = img[:,:,sid]
                        elif plane==1:
                            img_slice = img[:,:,:,sid]
                        elif plane==2:
                            img_slice = img[:,:,:,:,sid]
                        newprom = prom + [{'content': [view_prompt], 'mask': 1}]

                        valid_data_unit.append({
                            'imgs': img_slice,
                            'prompts': newprom,
                            'text': valid_data.gen_text_from_prompts(newprom),
                        })

                    train_imgs = torch.cat([i['imgs'] for i in valid_data_unit], dim=0).to(device=device, dtype=torch.float32)
                    train_tokens = tokenize([i['text'] for i in valid_data_unit], context_length=args['context_length']).to(device=device)
                    for i in range(train_imgs.shape[0]):
                        input_image = train_imgs[i].tile(train_imgs.shape[0],1,1,1)
                        mask = (((train_imgs>0)*(input_image<=0)+(train_imgs<=0)*(input_image>0))<0.5).to(device=device, dtype=torch.float32)
                        rec, _ = net.infer_syn(input_image, train_tokens)
                        psnr = torch_PSNR(train_imgs*mask, torch.clamp(rec, min=0)*mask, data_range=1)
                        test_psnr.append(psnr.item())
                #break
                
        loss = np.mean(test_psnr)
        recorder.update({'dis': np.mean(train_metrics['dis']), 'adv': np.mean(train_metrics['adv']), 'per': np.mean(train_metrics['per']), 'rec': np.mean(train_metrics['rec']), 'content': np.mean(train_metrics['content']), 'psnr': np.mean(test_psnr)})
        plotter.send(recorder.call())
        if best_loss<loss:
            best_loss = loss
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{},{},{},{}\n'.format(epoch+1, np.mean(train_metrics['rec']), np.mean(train_metrics['per']), np.mean(train_metrics['content']), np.mean(test_psnr)))
        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
        model_ema.restore(net.parameters())
        torch.cuda.empty_cache()
                

def get_args():
    parser = argparse.ArgumentParser('MAE clip pre-training', add_help=False)
    parser.add_argument('-cd', '--config_data', default='config/eva/dataset.yaml',
                        help='config file for dataset')
    parser.add_argument('-cm', '--config_model', default='config/eva/eva.yaml',
                        help='config file for dataset')
    parser.add_argument('-ct', '--config_train', default='config/eva/train.yaml',
                        help='config file for training')
    parser.add_argument('-d', '--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('-lg', '--load_gen', type=str, default=None,
                        help='Load generator model from a .pth file')
    parser.add_argument('-tf', '--test_fold', type=int, default=0,
                        help='test fold')
    parser.add_argument('-vf', '--valid_fold', type=int, default=1,
                        help='valid fold')
    parser.add_argument('-sd', '--start_dis', type=int, default=5,
                        help='start dis')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    
    with open(args.config_data, 'r') as f:
        config_data = yaml.safe_load(f)
    with open(args.config_model, 'r') as f:
        config_model = yaml.safe_load(f)
    with open(args.config_train, 'r') as f:
        config_train = yaml.safe_load(f)
        config_train['context_length'] = config_model['text_encoder']['context_length']
        config_train['test_fold'] = args.test_fold
        config_train['valid_fold'] = args.valid_fold
        config_train['start_dis'] = args.start_dis

    config_train['pretrain_eva']['ckpt'] = os.path.join(config_train['pretrain_eva']['ckpt'], str(args.test_fold))
    dir_checkpoint = config_train['pretrain_eva']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    config_train['pretrain_eva']['vis'] = os.path.join(config_train['pretrain_eva']['vis'], str(args.test_fold))
    dir_visualize = config_train['pretrain_eva']['vis']
    if not os.path.exists(dir_visualize):
        os.makedirs(dir_visualize)

    device = torch.device(args.device)
    logging.info(f'Using device {device}')

    net = EVA(config_model)
    net.to(device=device)
    print(net)

    if args.load_gen:
        #load_dict = torch.load(args.load, map_location=device)
        #net.load_state_dict(load_dict)
        net = load_weights(net, args.load_gen, device=device)
        print('[*] Load model from', args.load_gen)

    netd = NLayerDiscriminator(config_model['discriminator'])
    netd.to(device=device)
    print(netd)

    if args.load_gen:
        dis_path = os.path.join(os.path.dirname(args.load_gen), 'ckpt_tmp_d.pth')
        if os.path.exists(dis_path):
            netd = load_weights(netd, dis_path, device=device)
            print('[*] Load model from', dis_path)
    
    try:
        train(
            config_data,
            config_train,
            net=net,
            netd=netd,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
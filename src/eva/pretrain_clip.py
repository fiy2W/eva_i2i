import argparse
import logging
import numpy as np
import os
import yaml
import math
import random
from tqdm import tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision

import sys
sys.path.append('./src/')
from eva.model.eva import EVA
from eva.model.ema import EMAModel
from eva.model.text_encoder import tokenize
from eva.dataset.dataloader import Dataset_ALL
from eva.dataset.augmentation import random_aug
from eva.utils import Recorder, Plotter, load_weights
from eva.losses import SupConLoss

  
def train(args_data, args, net, device):
    # fix the seed for reproducibility
    seed = args['pretrain_clip']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = Dataset_ALL(args_data, mode='train', test_fold=args['test_fold'], valid_fold=args['valid_fold'])

    n_train = len(train_data)
    label_counts = Counter(train_data.dataset_label)
    class_weights = {label: 1.0 / count for label, count in label_counts.items()}
    weights = [class_weights[train_data.dataset_label[i]] for i in range(n_train)]
    weights = torch.DoubleTensor(weights)

    train_loader = DataLoader(train_data, batch_size=args['pretrain_clip']['batch_size'], num_workers=args['pretrain_clip']['num_workers'], pin_memory=True,
        sampler=WeightedRandomSampler(weights, n_train), collate_fn=lambda x: x)

    size = args_data['prep']['size']
    batch_size = args['pretrain_clip']['batch_size']
    epochs = args['pretrain_clip']['epochs']
    lr = np.float32(args['pretrain_clip']['lr'])
    dir_visualize = args['pretrain_clip']['vis']
    dir_checkpoint = args['pretrain_clip']['ckpt']
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    contrastive_loss = SupConLoss()
    n_step_pre_epoch = n_train//batch_size

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=args['pretrain_clip']['weight_decay'])
    model_ema = EMAModel(net.parameters())
    model_ema.to(device=device)

    recorder = Recorder(['clip'])
    plotter = Plotter(dir_visualize, keys1=['clip'])
    
    with open(os.path.join(dir_checkpoint, 'log.csv'), 'w') as f:
        f.write('epoch,clip\n')

    total_step = 0
    best_loss = 100
    for epoch in range(epochs):
        net.train()
        train_metrics = {
            'total': [], 'clip': [],
        }
        with tqdm(total=n_step_pre_epoch*args['pretrain_clip']['repeat_sample'], desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            #loader = iter(train_loader)
            for batch in train_loader:

                data_unit = []

                #batch = next(loader)
                for batch_1 in batch:
                    box_b = batch_1['box']
                    for batch_i in batch_1['flag']:
                        batch_i = int(batch_i)
                        data_unit.append({
                            'imgs': batch_1[batch_i].unsqueeze(0),
                            'prompts': batch_1['prompt'][batch_i],
                            'boxes': box_b,
                            'dataset': batch_1['dataset'],
                            'organ': batch_1['organ'],
                        })

                for _ in range(args['pretrain_clip']['repeat_sample']):
                    train_data_unit = []
                    
                    plane_dict = ['axial', 'coronal', 'sagittal']
                    for plane in range(3):
                        view_prompt = ' The image is visualized in the {} plane.'.format(plane_dict[plane])
                        for _ in range(args['pretrain_clip']['repeat_slice']):
                            for unit in data_unit:
                                img = unit['imgs']
                                prom = unit['prompts'].copy()
                                box = unit['boxes']

                                sid = random.randint(box[0+plane*2], box[1+plane*2])
                                sid = max(min(sid, size[plane]-1), 0)
                                if plane==0:
                                    img_slice = img[:,:,sid]
                                elif plane==1:
                                    img_slice = img[:,:,:,sid]
                                elif plane==2:
                                    img_slice = img[:,:,:,:,sid]
                                
                                newprom = prom + [{'content': [view_prompt], 'mask': 1}]

                                train_data_unit.append({
                                    'imgs': img_slice,
                                    'prompts': newprom,
                                    'text': train_data.gen_text_from_prompts(newprom),
                                })
                    
                    random.shuffle(train_data_unit)

                    N = min(args['pretrain_clip']['clip_num'], len(train_data_unit))
                    train_imgs = [train_data_unit[i]['imgs'] for i in range(N)]
                    train_prompts = [train_data_unit[i]['text'] for i in range(N)]
                    contrast_mask = np.eye(N)
                    for i in range(N):
                        text = train_prompts[i]
                        for j in range(N):
                            if i==j:
                                continue
                            anchor_prompts = train_data_unit[j]['prompts']
                            if train_data.is_subset_prompt(text, anchor_prompts):
                                contrast_mask[i, j] = 1

                    train_imgs = random_aug(torch.cat(train_imgs, dim=0).to(device=device, dtype=torch.float32))
                    train_tokens = tokenize(train_prompts, context_length=args['context_length']).to(device=device)
                    train_contrast_mask = torch.from_numpy(np.array(contrast_mask, dtype=np.int64)).to(device=device)

                    feat_imgs, feat_prompt = net(train_imgs, train_imgs, train_tokens, train_tokens, recon=False)
                    loss_clip = contrastive_loss(torch.stack([feat_imgs, feat_prompt], dim=1), mask=train_contrast_mask)
                    loss = loss_clip
                    if not math.isfinite(loss.item()):
                        print("Loss is {}, stopping training".format(loss.item()))
                        sys.exit(1)

                    optimizer.zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, norm_type=2)
                    optimizer.step()
                    model_ema.step(net.parameters())

                    train_metrics['total'].append(loss.item())
                    train_metrics['clip'].append(loss_clip.item())
                    pbar.set_postfix(**{'clip': loss_clip.item()})
                    pbar.update(1)

                    if (total_step % args['pretrain_clip']['view_freq']) == 0:
                        torchvision.utils.save_image(train_imgs, os.path.join(dir_visualize, '{}.jpg'.format(epoch)))
                    
                    if (total_step % args['pretrain_clip']['save_freq']) == 0:
                        model_ema.store(net.parameters())
                        model_ema.copy_to(net.parameters())
                        torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_tmp.pth'))
                        model_ema.restore(net.parameters())

                    total_step += 1
                #break
                #torch.cuda.empty_cache()
        
        loss = np.mean(train_metrics['clip'])
        recorder.update({'clip': np.mean(train_metrics['clip'])})
        plotter.send(recorder.call())
        if best_loss>loss:
            best_loss = loss
            model_ema.store(net.parameters())
            model_ema.copy_to(net.parameters())
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_best.pth'))
            model_ema.restore(net.parameters())
            with open(os.path.join(dir_checkpoint, 'log.csv'), 'a+') as f:
                f.write('{},{}\n'.format(epoch+1, np.mean(train_metrics['clip'])))
        #torch.save(net.state_dict(), os.path.join(dir_checkpoint, 'ckpt_latest.pth'))
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

    config_train['pretrain_clip']['ckpt'] = os.path.join(config_train['pretrain_clip']['ckpt'], str(args.test_fold))
    dir_checkpoint = config_train['pretrain_clip']['ckpt']
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    
    config_train['pretrain_clip']['vis'] = os.path.join(config_train['pretrain_clip']['vis'], str(args.test_fold))
    dir_visualize = config_train['pretrain_clip']['vis']
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
    
    try:
        train(
            config_data,
            config_train,
            net=net,
            device=device,
        )
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
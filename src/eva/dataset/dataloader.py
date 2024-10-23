import os
import numpy as np
import json
import SimpleITK as sitk
import random
from scipy.ndimage import zoom
import math
import re
from copy import deepcopy

import torch
from torch.utils.data import Dataset

import sys
sys.path.append('./src/')

from eva.dataset.prompt_sample import *


class Dataset_ALL(Dataset):
    def __init__(self, args, mode='train', test_fold=0, valid_fold=1):
        self.mode = mode
        self.size = args['prep']['size']
        self.box_percentile = args['prep']['box_percentile']
        self.max_out_seq = args['prep']['max_out_seq']
        self.image_root = {}
        self.aid_list = []
        self.dataset_label = []
        
        dataset_label = 0
        for dataset_name in args['dataset']:
            self.image_root[dataset_name] = args[dataset_name]['image']
            list_root = args[dataset_name]['list']
            dict_path = args[dataset_name]['dict']

            with open(dict_path, 'r') as f:
                load_dict = json.load(f)

            aid_list = []
            if self.mode=='test':
                with open(os.path.join(list_root, 'fold_{}.csv'.format(test_fold)), 'r') as f:
                    strs = f.readlines()
                    aid_list = [i.split('\n')[0] for i in strs]
            elif self.mode=='valid':
                with open(os.path.join(list_root, 'fold_{}.csv'.format(valid_fold)), 'r') as f:
                    strs = f.readlines()
                    aid_list = [i.split('\n')[0] for i in strs]
            elif self.mode=='train':
                for i in range(5):
                    if i==test_fold or i==valid_fold:
                        continue
                    with open(os.path.join(list_root, 'fold_{}.csv'.format(i)), 'r') as f:
                        strs = f.readlines()
                        aid_list += [i.split('\n')[0] for i in strs]
            elif 'all' in self.mode:
                for i in range(5):
                    with open(os.path.join(list_root, 'fold_{}.csv'.format(i)), 'r') as f:
                        strs = f.readlines()
                        aid_list += [i.split('\n')[0] for i in strs]
            
            self.aid_list += [{'name': name, 'dataset': dataset_name, 'info': load_dict[name]} for name in aid_list]
            self.dataset_label += [dataset_label for name in aid_list]
            dataset_label += 1
    
    def __len__(self):
        return len(self.aid_list)
    
    def __getitem__(self, index):
        x_idx = self.aid_list[index]
        dataset_id = self.dataset_label[index]
        dataset = x_idx['dataset']
        image_root = self.image_root[dataset]
        name = x_idx['name']
        info = deepcopy(x_idx['info'])

        imgs, flag, info_out, box = self.load_images(image_root, name, info, dataset)
        if 'train' in self.mode:
            random.shuffle(flag)
            flag = flag[:self.max_out_seq]
            flag.sort()
        N = len(imgs)
        output = {
            'N': N,
            'flag': flag,
            'prompt': {},
            'dataset': dataset_id,
            'info': info_out,
            'organ': info_out[0]['Organ'],
        }
        
        seed = random.random()
        for i in range(N):
            imgs_prep, crop_box = self.preprocess(imgs[i], box, seed)
            output[i] = torch.from_numpy(imgs_prep)
            output['prompt'][i] = self.gen_prompt(info_out[i])
        output['box'] = crop_box

        return output
    
    def gen_prompt(self, info):
        if len(info.keys())==0:
            return []
        
        organ = info['Organ']
        if organ=='brain':
            info['Organ'] = [organ, 'head']
        elif organ in ['pelvis', 'spine']:
            info['Organ'] = [organ, 'abdomen']
        elif organ in ['prostate']:
            info['Organ'] = [organ, 'abdomen', 'pelvis']
        else:
            info['Organ'] = [organ]
        
        prompt = prompt_overview(info)
        prompt += prompt_ct(info)
        if info['Modality']=='magnetic resonance imaging':
            prompt += prompt_mr(info)
        prompt += prompt_preprocess(info)
        
        return prompt
    
    def gen_text_from_prompts(self, prompts):
        text = ''
        for p in prompts:
            t = random.choice(p['content'])
            if p['mask']==1:
                text += t
            else:
                if self.mode=='valid' or random.random()>0.5:
                    text += t
        return text
    
    def is_subset_prompt(self, text, anchor):
        anchor_mask = [m['mask'] for m in anchor]
        anchor_use = [0 for _ in anchor]
        texts = text.split('. ')
        text_exist = [0 for _ in texts]
        for i, t in enumerate(texts):
            if i>0:
                t = ' '+t
            if t[-1]!='.':
                t += '.'
            for j, ju in enumerate(anchor_use):
                if ju==1:
                    continue
                if t in anchor[j]['content']:
                    anchor_use[j] = 1
                    text_exist[i] = 1
                    break
            if text_exist[i]!=1:
                return False
        for am, au in zip(anchor_mask, anchor_use):
            if am==1 and au==0:
                return False
        return True

    
    def pad(self, arr):
        d, w, h = arr.shape[-3:]
        s1, s2, s3 = self.size
        s1 = max(d, s1)
        s2 = max(w, s2)
        s3 = max(h, s3)

        pd = (s1-d)//2 if s1>=d else 0
        pw = (s2-w)//2 if s2>=w else 0
        ph = (s3-h)//2 if s3>=h else 0

        if len(arr.shape)==3:
            arr = np.pad(arr, [[pd,s1-d-pd],[pw,s2-w-pw],[ph,s3-h-ph]])
        elif len(arr.shape)==4:
            arr = np.pad(arr, [[0,0],[pd,s1-d-pd],[pw,s2-w-pw],[ph,s3-h-ph]])
        
        return arr
    
    def norm_ct(self, arr, mask):
        amin = -1024
        amax = amin + 3000
        arr = (np.clip(arr, amin, a_max=None) - amin) / (amax - amin)#(np.clip(arr, amin, amax) - amin) / (amax - amin) # 0-1
        arr = self.pad(arr)
        arr = arr*mask
        return arr
    
    def synthrad_norm_cbct(self, arr, mask):
        amin = np.percentile(arr, 0.5)
        if amin>=0:
            amin = 0
        elif amin<-500:
            amin = -1024
        else:
            raise ValueError('Unknown min value {}'.format(amin))

        amax = amin + 1500
        arr = (np.clip(arr, amin, a_max=None) - amin) / (amax - amin)#(np.clip(arr, amin, amax) - amin) / (amax - amin) # 0-1
        arr = self.pad(arr)
        arr = arr*mask
        return arr
    
    def norm_mr(self, arr, mask):
        amin = 0
        amax = max(np.percentile(arr, 99.5), 1)
        arr = (np.clip(arr, amin, a_max=None) - amin) / (amax - amin)#(np.clip(arr, amin, amax) - amin) / (amax - amin) # 0-1
        arr = self.pad(arr)
        arr = arr*mask
        return arr
    
    def preprocess(self, x, box, seed):
        x, crop_box = self.crop(x, box, seed)

        x = np.expand_dims(x, axis=0) # [1,d,w,h]
        return x, crop_box
    
    def box(self, mask):
        md = np.sum(mask, axis=(1,2))
        mw = np.sum(mask, axis=(0,2))
        mh = np.sum(mask, axis=(0,1))

        rmd = np.where(md>1)[0]
        rmw = np.where(mw>1)[0]
        rmh = np.where(mh>1)[0]
        return [
            np.int32(np.percentile(rmd, self.box_percentile)),
            np.int32(np.percentile(rmd, 100-self.box_percentile)),
            np.int32(np.percentile(rmw, self.box_percentile)),
            np.int32(np.percentile(rmw, 100-self.box_percentile)),
            np.int32(np.percentile(rmh, self.box_percentile)),
            np.int32(np.percentile(rmh, 100-self.box_percentile)),
        ]
    
    def crop(self, arr, box, seed):
        d, w, h = arr.shape[-3:]
        s1, s2, s3 = self.size
        crop_box = []

        if 'train' in self.mode:
            if box[1]-box[0]>=s1:
                d1, d2 = box[0], box[1]-s1
                crop_box.extend([0, s1-1])
            else:
                d1, d2 = max(0, box[1]-s1), min(box[0], d-s1-1)
                crop_box.extend([box[0]-d1, box[1]-d1])

            if box[3]-box[2]>=s2:
                w1, w2 = box[2], box[3]-s2
                crop_box.extend([0, s2-1])
            else:
                w1, w2 = max(0, box[3]-s2), min(box[2], w-s2-1)
                crop_box.extend([box[2]-w1, box[3]-w1])

            if box[5]-box[4]>=s3:
                h1, h2 = box[4], box[5]-s3
                crop_box.extend([0, s3-1])
            else:
                h1, h2 = max(0, box[5]-s3), min(box[4], h-s3-1)
                crop_box.extend([box[4]-h1, box[5]-h1])

            random.seed(seed)
            rd = random.randint(d1, d2) if s1<d else 0
            rw = random.randint(w1, w2) if s2<w else 0
            rh = random.randint(h1, h2) if s3<h else 0
        else:
            rd = (d-s1)//2 if s1<d else 0
            rw = (w-s2)//2 if s2<w else 0
            rh = (h-s3)//2 if s3<h else 0
            
        arr = arr[...,rd:rd+s1,rw:rw+s2,rh:rh+s3]
        return arr, crop_box
    
    def load_images(self, image_root, name, info, dataset):
        if dataset=='synthrad':
            organ = info['computed tomography']['Organ']
            mask_path = os.path.join(image_root, 'Task{}'.format(name[0]), organ, name, 'mask.nii.gz')
            ct_path = os.path.join(image_root, 'Task{}'.format(name[0]), organ, name, 'ct.nii.gz')
            mr_path = os.path.join(image_root, 'Task{}'.format(name[0]), organ, name, 'mr.nii.gz')
            cbct_path = os.path.join(image_root, 'Task{}'.format(name[0]), organ, name, 'cbct.nii.gz')
            mr_sk_path = os.path.join(image_root, 'Task{}'.format(name[0]), organ, name, 'mr_skull_removal.nii.gz')
            mask = self.pad(sitk.GetArrayFromImage(sitk.ReadImage(mask_path)))
            ct = self.norm_ct(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), mask)
            mr = self.norm_mr(sitk.GetArrayFromImage(sitk.ReadImage(mr_path)), mask) if 'magnetic resonance imaging' in info else np.zeros_like(mask)
            cbct = self.synthrad_norm_cbct(sitk.GetArrayFromImage(sitk.ReadImage(cbct_path)), mask) if 'cone beam computed tomography' in info else np.zeros_like(mask)
            mr_sk = self.norm_mr(sitk.GetArrayFromImage(sitk.ReadImage(mr_sk_path)), mask)+1e-9 if 'MRI skull removal' in info else np.zeros_like(mask)

            imgs = [ct, mr, mr_sk, cbct]
            flag = [0, 1, 2] if 'magnetic resonance imaging' in info else [0, 3]
            info_out = [info['computed tomography'],
                        info['magnetic resonance imaging'] if 'magnetic resonance imaging' in info else {'Organ': info['computed tomography']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted', 'Contrast': 'Gadolinium', 'Spacing': info['computed tomography']['Spacing']},
                        info['MRI skull removal'] if 'MRI skull removal' in info else {'Organ': info['computed tomography']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted', 'Contrast': 'Gadolinium', 'Preprocess': ['skull removal'], 'Spacing': info['computed tomography']['Spacing']},
                        info['cone beam computed tomography'] if 'cone beam computed tomography' in info else {'Organ': info['computed tomography']['Organ'], 'Modality': 'cone beam computed tomography', 'Spacing': info['computed tomography']['Spacing']}]
            
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='crossmoda':
            sequence = list(info.keys())[0]
            mr_path = os.path.join(image_root, name)
            img_arr = sitk.GetArrayFromImage(sitk.ReadImage(mr_path))
            mask = self.pad(img_arr>0)
            if sequence=='T1-weighted':
                cet1 = self.norm_mr(img_arr, mask)
                hrt2 = np.zeros_like(mask)
                flag = [0]
                info_out = [
                    info['T1-weighted'],
                    {'Organ': info['T1-weighted']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T2-weighted', 'Spacing': info['T1-weighted']['Spacing']},
                ]
            else:
                hrt2 = self.norm_mr(img_arr, mask)
                cet1 = np.zeros_like(mask)
                flag = [1]
                info_out = [
                    {'Organ': info['T2-weighted']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted', 'Contrast': 'Gadolinium', 'Spacing': info['T2-weighted']['Spacing']},
                    info['T2-weighted'],
                ]

            imgs = [cet1, hrt2]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='brats':
            t1_path = os.path.join(image_root, name, '{}_t1.nii.gz'.format(os.path.basename(name)))
            t1ce_path = os.path.join(image_root, name, '{}_t1ce.nii.gz'.format(os.path.basename(name)))
            t2_path = os.path.join(image_root, name, '{}_t2.nii.gz'.format(os.path.basename(name)))
            flair_path = os.path.join(image_root, name, '{}_flair.nii.gz'.format(os.path.basename(name)))

            t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            t1ce_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1ce_path))
            t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
            flair_arr = sitk.GetArrayFromImage(sitk.ReadImage(flair_path))
            mask = self.pad(((t1_arr>0)+(t1ce_arr>0)+(t2_arr>0)+(flair_arr>0))>0.5)

            t1 = self.norm_mr(t1_arr, mask)
            t1ce = self.norm_mr(t1ce_arr, mask)
            t2 = self.norm_mr(t2_arr, mask)
            flair = self.norm_mr(flair_arr, mask)

            imgs = [t1, t1ce, t2, flair]
            flag = [0, 1, 2, 3]
            info_out = [
                info['T1-weighted'],
                info['T1-contrast'],
                info['T2-weighted'],
                info['T2-FLAIR'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='duke-breast-cancer-mri':
            n_dce = len(list(info.keys()))
            if 'T1-weighted' in info:
                n_dce -= 1
                t1_path = os.path.join(image_root, name, 't1.nii.gz')
            dce_path = os.path.join(image_root, name, 'dce{}.nii.gz')

            dces = []
            imgs = []
            flag = []
            info_out = []
            for i in range(n_dce):
                dce_arr = sitk.GetArrayFromImage(sitk.ReadImage(dce_path.format(i)))
                dces.append(dce_arr)
                if i==0:
                    mask = self.pad(dce_arr>0)
            dces = np.stack(dces, axis=0)
            dces = self.norm_mr(dces, np.expand_dims(mask, axis=0))
            for i in range(n_dce):
                imgs.append(dces[i])
                flag.append(i)
                info_out.append(info['DCE-{}'.format(i)])

            if 'T1-weighted' in info:
                t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
                t1 = self.norm_mr(t1_arr, mask)
                imgs.append(t1)
                flag.append(n_dce)
                info_out.append(info['T1-weighted'])
            else:
                imgs.append(np.zeros_like(mask))
                info_out.append({'Organ': info['DCE-0']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted', 'Spacing': info['DCE-0']['Spacing']},)
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='chaos':
            if 'CT' in info:
                ct_path = os.path.join(image_root, name, 'ct.nii.gz')
                ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
                mask = self.pad(ct_arr>-1024)
                ct_arr = self.norm_ct(ct_arr, mask)
                imgs = [ct_arr, np.zeros_like(ct_arr), np.zeros_like(ct_arr), np.zeros_like(ct_arr)]
                flag = [0]
                info_out = [
                    info['CT'],
                    {'Organ': info['CT']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted dual-echo in-phase', 'Spacing': info['CT']['Spacing']},
                    {'Organ': info['CT']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted dual-echo out-of-phase', 'Spacing': info['CT']['Spacing']},
                    {'Organ': info['CT']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T2-weighted spectral presaturation with inversion recovery', 'ScanOptions': 'fat saturation', 'Spacing': info['CT']['Spacing']},
                ]
                box = self.box(mask)
                return imgs, flag, info_out, box
            else:
                inphase_path = os.path.join(image_root, name, 't1inphase.nii.gz')
                outphase_path = os.path.join(image_root, name, 't1outphase.nii.gz')
                t2_path = os.path.join(image_root, name, 't2.nii.gz')

                inphase_arr = sitk.GetArrayFromImage(sitk.ReadImage(inphase_path))
                outphase_arr = sitk.GetArrayFromImage(sitk.ReadImage(outphase_path))
                t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
                mask = self.pad(((inphase_arr>0)+(outphase_arr>0)+(t2_arr>0))>0.5)
                inphase_arr = self.norm_mr(inphase_arr, mask)
                outphase_arr = self.norm_mr(outphase_arr, mask)
                t2_arr = self.norm_mr(t2_arr, mask)
                imgs = [np.zeros_like(inphase_arr), inphase_arr, outphase_arr, t2_arr]
                flag = [1,2,3]
                info_out = [
                    {'Organ': info['T2']['Organ'], 'Modality': 'computed tomography', 'Contrast': 'unknown', 'Spacing': info['T2']['Spacing']},
                    info['T1-inphase'],
                    info['T1-outphase'],
                    info['T2'],
                ]
                box = self.box(mask)
                return imgs, flag, info_out, box
        
        elif dataset=='amos':
            if 'computed tomography' in info:
                ct_path = os.path.join(image_root, name+'.nii.gz')
                ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
                mask = self.pad(ct_arr>-1024)
                ct_arr = self.norm_ct(ct_arr, mask)
                flag = [0]
                imgs = [ct_arr, np.zeros_like(ct_arr)]
                info_out = [
                    info['computed tomography'],
                    {'Organ': info['computed tomography']['Organ'], 'Modality': 'magnetic resonance imaging', 'Sequence': 'T1-weighted', 'Spacing': info['computed tomography']['Spacing']},
                ]
            else:
                mr_path = os.path.join(image_root, name+'.nii.gz')
                mr_arr = sitk.GetArrayFromImage(sitk.ReadImage(mr_path))
                mask = self.pad(mr_arr>0)
                mr_arr = self.norm_mr(mr_arr, mask)
                flag = [1]
                imgs = [np.zeros_like(mr_arr), mr_arr]
                info_out = [
                    {'Organ': info['magnetic resonance imaging']['Organ'], 'Modality': 'computed tomography', 'Spacing': info['magnetic resonance imaging']['Spacing']},
                    info['magnetic resonance imaging'],
                ]

            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='pancreas-ct':
            ct_path = os.path.join(image_root, name)
            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            mask = self.pad(ct_arr>-1024)
            ct_arr = self.norm_ct(ct_arr, mask)
            flag = [0]
            imgs = [ct_arr]
            info_out = [
                info['CT'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='mrspineseg':
            mr_path = os.path.join(image_root, name)
            mr_arr = sitk.GetArrayFromImage(sitk.ReadImage(mr_path))
            mask = self.pad(mr_arr>0)
            mr_arr = self.norm_mr(mr_arr, mask)
            flag = [0]
            imgs = [mr_arr]
            info_out = [
                info['MR'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='ixi':
            t1_path = os.path.join(image_root, name, 'IXI-T1.nii.gz')
            t1_sk_path = os.path.join(image_root, name, 'IXI-T1_skull_rm.nii.gz')
            t2_path = os.path.join(image_root, name, 'IXI-T2.nii.gz')
            pd_path = os.path.join(image_root, name, 'IXI-PD.nii.gz')
            mra_path = os.path.join(image_root, name, 'IXI-MRA.nii.gz')

            t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            t1_sk_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_sk_path))
            t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path)) if os.path.exists(t2_path) else np.zeros_like(t1_arr)
            pd_arr = sitk.GetArrayFromImage(sitk.ReadImage(pd_path)) if os.path.exists(pd_path) else np.zeros_like(t1_arr)
            mra_arr = sitk.GetArrayFromImage(sitk.ReadImage(mra_path)) if os.path.exists(mra_path) else np.zeros_like(t1_arr)
            mask = self.pad(((t1_arr>0)+(t2_arr>0)+(pd_arr>0)+(mra_arr>0))>0.5)

            t1 = self.norm_mr(t1_arr, mask)
            t1_sk = self.norm_mr(t1_sk_arr, mask)
            skull = t1 - t1_sk
            t1_sk += ((skull>0)*(t1>0)*1e-9)
            t2 = self.norm_mr(t2_arr, mask)
            t2_sk = self.norm_mr(t2_arr*(t1_sk_arr>0), mask)
            t2_sk += ((skull>0)*(t2>0)*1e-9)
            pd = self.norm_mr(pd_arr, mask)
            pd_sk = self.norm_mr(pd_arr*(t1_sk_arr>0), mask)
            pd_sk += ((skull>0)*(pd>0)*1e-9)
            mra = self.norm_mr(mra_arr, mask)
            mra_sk = self.norm_mr(mra_arr*(t1_sk_arr>0), mask)
            mra_sk += ((skull>0)*(mra>0)*1e-9)

            imgs = [t1, t1_sk, t2, t2_sk, pd, pd_sk, mra, mra_sk]
            flag = [0,1]
            if os.path.exists(t2_path):
                flag.extend([2,3])
            if os.path.exists(pd_path):
                flag.extend([4,5])
            if os.path.exists(mra_path):
                flag.extend([6,7])
            info_out = [
                info['IXI-T1'],
                info['IXI-T1_skull_rm'],
                info['IXI-T2'] if os.path.exists(t2_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'T2-weighted', 'Spacing': info['IXI-T1']['Spacing']},
                info['IXI-T2_skull_rm'] if os.path.exists(t2_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'T2-weighted', 'Preprocess': ['skull removal'], 'Spacing': info['IXI-T1']['Spacing']},
                info['IXI-PD'] if os.path.exists(pd_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'proton density weighted', 'Spacing': info['IXI-T1']['Spacing']},
                info['IXI-PD_skull_rm'] if os.path.exists(pd_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'proton density weighted', 'Preprocess': ['skull removal'], 'Spacing': info['IXI-T1']['Spacing']},
                info['IXI-MRA'] if os.path.exists(mra_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'magnetic resonance angiography', 'Spacing': info['IXI-T1']['Spacing']},
                info['IXI-MRA_skull_rm'] if os.path.exists(mra_path) else {'Organ': 'brain', 'Modality': 'magnetic resonance imaging', 'Sequence': 'magnetic resonance angiography', 'Preprocess': ['skull removal'], 'Spacing': info['IXI-T1']['Spacing']},
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='promise12':
            mr_path = os.path.join(image_root, name)
            mr_arr = sitk.GetArrayFromImage(sitk.ReadImage(mr_path))
            mask = self.pad(mr_arr>0)
            mr_arr = self.norm_mr(mr_arr, mask)
            flag = [0]
            imgs = [mr_arr]
            info_out = [
                info['MR'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='prostate-3t':
            mr_path = os.path.join(image_root, name)
            mr_arr = sitk.GetArrayFromImage(sitk.ReadImage(mr_path))
            mask = self.pad(mr_arr>0)
            mr_arr = self.norm_mr(mr_arr, mask)
            flag = [0]
            imgs = [mr_arr]
            info_out = [
                info['MR'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='i2cvb':
            n_dce = len(list(info.keys()))
            if 'T2W' in info:
                n_dce -= 1
                t2_path = os.path.join(image_root, name, 'T2W.nii.gz')
            dce_path = os.path.join(image_root, name, 'DCE-{}.nii.gz')

            dces = []
            imgs = []
            flag = []
            info_out = []
            for i in range(n_dce):
                dce_arr = sitk.GetArrayFromImage(sitk.ReadImage(dce_path.format(i)))
                dces.append(dce_arr)
                if i==0:
                    mask = self.pad(dce_arr>0)
            if n_dce>0:
                dces = np.stack(dces, axis=0)
                dces = self.norm_mr(dces, np.expand_dims(mask, axis=0))
            for i in range(n_dce):
                imgs.append(dces[i])
                flag.append(i)
                info_out.append(info['DCE-{}'.format(i)])

            if 'T2W' in info:
                t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
                if n_dce==0:
                    mask = self.pad(t2_arr>0)
                t2 = self.norm_mr(t2_arr, mask)
                imgs.append(t2)
                flag.append(n_dce)
                info_out.append(info['T2W'])
            else:
                imgs.append(np.zeros_like(mask))
                info_out.append({'Organ': 'prostate', 'Modality': 'magnetic resonance imaging', 'Sequence': 'T2-weighted', 'Spacing': info['DCE-0']['Spacing']},)
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='prostate-diagnosis':
            n_dce = len(list(info.keys()))
            if 'T2WTSEAX' in info:
                n_dce -= 1
                t2_path = os.path.join(image_root, name, 'T2WTSEAX.nii.gz')
            if 'T1WTSEAX' in info:
                n_dce -= 1
                t1_path = os.path.join(image_root, name, 'T1WTSEAX.nii.gz')
            if 'T2WTSECOR' in info:
                n_dce -= 1
                t2c_path = os.path.join(image_root, name, 'T2WTSECOR.nii.gz')
            dce_path = os.path.join(image_root, name, 'DCE-{}.nii.gz')

            dces = []
            imgs = []
            flag = []
            info_out = []
            for i in range(n_dce):
                dce_arr = sitk.GetArrayFromImage(sitk.ReadImage(dce_path.format(i)))
                dces.append(dce_arr)
                if i==0:
                    mask = self.pad(dce_arr>0)
            if n_dce>0:
                dces = np.stack(dces, axis=0)
                dces = self.norm_mr(dces, np.expand_dims(mask, axis=0))
            for i in range(n_dce):
                imgs.append(dces[i])
                flag.append(i)
                info_out.append(info['DCE-{}'.format(i)])

            if 'T2WTSEAX' in info:
                t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
                if n_dce==0:
                    mask = self.pad(t2_arr>0)
                t2 = self.norm_mr(t2_arr, mask)
                imgs.append(t2)
                flag.append(0 if len(flag)==0 else flag[-1]+1)
                info_out.append(info['T2WTSEAX'])
            if 'T1WTSEAX' in info:
                t1_arr = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
                t1 = self.norm_mr(t1_arr, mask)
                imgs.append(t1)
                flag.append(0 if len(flag)==0 else flag[-1]+1)
                info_out.append(info['T1WTSEAX'])
            if 'T2WTSECOR' in info:
                t2c_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2c_path))
                t2c = self.norm_mr(t2c_arr, mask)
                imgs.append(t2c)
                flag.append(0 if len(flag)==0 else flag[-1]+1)
                info_out.append(info['T2WTSECOR'])
            
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='prostate-mri':
            t2a_path = os.path.join(image_root, name, 'T2 TSE ax.nii.gz')
            t2s_path = os.path.join(image_root, name, 'T2 TSE sag.nii.gz')
            t2c_path = os.path.join(image_root, name, 'T2 TSE cor.nii.gz')
            t2a_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2a_path))
            t2s_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2s_path))
            t2c_arr = sitk.GetArrayFromImage(sitk.ReadImage(t2c_path))
            mask = self.pad(((t2a_arr>0)+(t2s_arr>0)+(t2c_arr>0))>0.5)
            t2a = self.norm_mr(t2a_arr, mask)
            t2s = self.norm_mr(t2s_arr, mask)
            t2c = self.norm_mr(t2c_arr, mask)
            flag = [0,1,2]
            imgs = [t2a, t2s, t2c]
            info_out = [
                info['T2 TSE ax'],
                info['T2 TSE sag'],
                info['T2 TSE cor'],
            ]
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='oasis3':
            if 'ct' in info:
                ct_path = os.path.join(image_root, name, 'ct.nii.gz')
                ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
                mask = self.pad(ct_arr>-1024)
                ct = self.norm_ct(ct_arr, mask)
                flag = [0]
                imgs = [ct]
                info_out = [info['ct']]
            else:
                t1_path = os.path.join(image_root, name, 't1.nii.gz')
                t1_sk_path = os.path.join(image_root, name, 't1_skull_rm.nii.gz')
                t2_path = os.path.join(image_root, name, 't2.nii.gz')
                t2star_path = os.path.join(image_root, name, 't2star.nii.gz')
                tset2_path = os.path.join(image_root, name, 'tse_t2.nii.gz')
                flair_path = os.path.join(image_root, name, 'flair.nii.gz')
                swi_path = os.path.join(image_root, name, 'swi.nii.gz')
                flag = []
                imgs = []
                info_out = []
                mask = 0
                flag_id = 0
                for p in [t1_path, t1_sk_path]:
                    if not os.path.exists(p):
                        continue
                    arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
                    mask += self.pad(arr>0)
                    flag.append(flag_id)
                    imgs.append(arr)
                    info_out.append(info[os.path.basename(p).split('.')[0]])
                    flag_id += 1
                with_t1 = len(flag)==2
                for p in [t2_path, t2star_path, tset2_path, flair_path, swi_path]:
                    if not os.path.exists(p):
                        continue
                    arr = sitk.GetArrayFromImage(sitk.ReadImage(p))
                    mask += self.pad(arr>0)
                    flag.append(flag_id)
                    imgs.append(arr)
                    info_out.append(info[os.path.basename(p).split('.')[0]])
                    flag_id += 1
                    if with_t1:
                        flag.append(flag_id)
                        imgs.append(arr*(imgs[1]>0))
                        info_out.append(info[os.path.basename(p).split('.')[0]+'_skull_rm'])
                        flag_id += 1

                mask = (mask>0.5)
                imgs = [self.norm_mr(arr, mask) for arr in imgs]
                if with_t1:
                    skull = imgs[0] - imgs[1]
                    imgs[1] += ((skull>0)*(imgs[0]>0)*1e-9)
                    for i in range(3, len(imgs), 2):
                        imgs[i] += ((skull>0)*(imgs[i-1]>0)*1e-9)
            
            box = self.box(mask)
            return imgs, flag, info_out, box
        
        elif dataset=='nki':
            imgs = []
            flag = []
            info_out = []
            aid, date = name.split('-')
            mask = None
            if 'dce-0' in info.keys():
                dces = []
                if os.path.exists(os.path.join(image_root, aid, date, 'dce')):
                    dce_fs = 'dce'
                else:
                    dce_fs = 'nfsdce'
                for i in range(2):
                    dce_path = os.path.join(image_root, aid, date, '{}/{}.nii.gz'.format(dce_fs, i+1))
                    if not os.path.exists(dce_path):
                        dce_path = os.path.join(image_root, aid, date, '{}/{}.nii.gz'.format(dce_fs, 1))
                    dce_arr = sitk.GetArrayFromImage(sitk.ReadImage(dce_path))
                    dces.append(dce_arr)
                    if i==0:
                        mask = self.pad(dce_arr>0)
                dces = np.stack(dces, axis=0)
                dces = self.norm_mr(dces, np.expand_dims(mask, axis=0))
                for i in range(2):
                    imgs.append(dces[i])
                    flag.append(i)
                    info_out.append(info['dce-{}'.format(i)])

            if 't2' in info:
                t2_arr = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(image_root, aid, date, 't2/t2.nii.gz')))
                if mask is None:
                    mask = self.pad(t2_arr>0)
                t2 = self.norm_mr(t2_arr, mask)
                imgs.append(t2)
                if len(flag)==0:
                    flag.append(0)
                else:
                    flag.append(flag[-1]+1)
                info_out.append(info['t2'])
            
            if 'dwi-0' in info.keys():
                dwis = []
                for i in [0, 150, 800, 1500]:
                    dwi_path = os.path.join(image_root, aid, date, 'dwi/{}.nii.gz'.format(i))
                    dwi_arr = sitk.GetArrayFromImage(sitk.ReadImage(dwi_path))
                    if mask is None:
                        mask = self.pad(dwi_arr>0)
                    dwis.append(dwi_arr)
                dwis = np.stack(dwis, axis=0)
                dwis = self.norm_mr(dwis, np.expand_dims(mask, axis=0))
                for i in range(4):
                    imgs.append(dwis[i])
                    if len(flag)==0:
                        flag.append(0)
                    else:
                        flag.append(flag[-1]+1)
                    info_out.append(info['dwi-{}'.format([0, 150, 800, 1500][i])])
            
            box = self.box(mask)
            return imgs, flag, info_out, box


if __name__ == '__main__':
    import yaml
    from torch.utils.data import DataLoader
    with open('config/eva/dataset.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    train_data = Dataset_ALL(config_data, mode='train_all')
    print(len(train_data))
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    for batch in train_loader:
        print(batch['prompt'])
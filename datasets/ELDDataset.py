import os
import re
import torch
import rawpy
import numpy as np
import exifread
from torch.utils.data import Dataset
import sys
sys.path.append('E:\Deep Learning\MetaRawFormer\MetaRawFormer')
#Notice: The ELD dataset is only for evaluation

from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class ELDDataset(Dataset):
    def __init__(self, data_dir, camera, pairs_file_path, metarange_file='metarange.txt', patch_size=None, split='train',
                transpose=False, h_flip=False, v_flip=False, **kwargs):
        super(ELDDataset, self).__init__()
        assert camera in ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
        self.data_dir = data_dir
        self.camera = camera
        self.split = split
        self.patch_size = patch_size
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.pairs_file_path=os.path.join(data_dir, pairs_file_path)
        self.metarange_file = os.path.join(data_dir, camera, metarange_file)
        #===================================================================
        #self.keys_min_max=[[1/3200, 30],[50, 25600],[16/5, 22],[21, 223]]
        self.keys_min_max = {}
        with open(self.metarange_file, 'r') as f:
            for i, key_min_max in enumerate(f):
                key_min_max = key_min_max.strip()
                key, _min, _max = key_min_max.split(' ')
                self.keys_min_max[key] = [eval(_min), eval(_max)]
        #===================================================================
        self.img_info=[]
        with open(self.pairs_file_path, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()
                input_path, gt_path, ratio = img_pair.split(' ')
                input_path = os.path.join(data_dir, self.camera, input_path)
                gt_path = os.path.join(data_dir, self.camera, gt_path)

                _id = os.path.basename(input_path)#10003_00_10s.ARW
                _id, extension = os.path.splitext(_id)#10003_00_10s    .ARW
                metainfo_path = os.path.join(self.data_dir, self.camera, _id + '.txt')
                meta = {}
                with open(metainfo_path, 'r') as f:
                    for i, k_v in enumerate(f):
                        k_v = k_v.strip()#"ISO": "200"
                        if k_v:
                            key, value = k_v.split(': ')
                            key = key.strip('"')
                            value = value.strip('"')
                            if key == 'ExposureTime':
                                value = float(eval(value))
                            elif key == 'FocalLength':
                                value = float(re.search(r'(\d+(\.\d+)?)', value).group(1))
                            elif key in ['BrightnessValue','ColorMatrix', 'BlueBalance', 'RedBalance', 'LightValue']:
                                continue
                            else:
                                value = eval(value)
                            meta[key] = value
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
                    'meta': meta,
                    'ratio': np.float32(ratio),
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']
        gt_path = info['gt_path']
        
        with rawpy.imread(input_path) as raw:
            input = self.pack_raw_bayer(raw) * info['ratio']

        with rawpy.imread(gt_path) as raw:
            gt = self.pack_raw_bayer(raw)
        
        
        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input = np.flip(input, axis=2)
                gt = np.flip(gt, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input = np.flip(input, axis=1)
                gt = np.flip(gt, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input = np.transpose(input, (0, 2, 1))
                gt = np.transpose(gt, (0, 2, 1)) 
            if self.patch_size:
                input, gt = self.crop_random_patch(input, gt, self.patch_size)
                input = input.copy()
                gt = gt.copy()
        
        input = np.ascontiguousarray(input)
        gt = np.ascontiguousarray(gt)
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)
        
        input_metainfo = self.getMetaInfoTensor(info['meta'])

        data = {
            'input_raw': input, 
            'gt_raw': gt, 
            'input_metainfo': input_metainfo,
            'input_path':input_path, 
            'gt_path': gt_path
        }
        
        return data

    def __len__(self):
        return len(self.img_info)
    
    def getMetaInfoTensor(self, metainfo):
        res = []
        for i, (k, v) in enumerate(metainfo.items()):
            min = self.keys_min_max[k][0]
            max = self.keys_min_max[k][1]
            res.append(torch.tensor(v/max, dtype=torch.float32))
        return torch.tensor(res)
    
    def crop_random_patch(self, input_raw, gt_raw, patch_size):
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]

        return input_raw, gt_raw
    
    def pack_raw_bayer(self, raw):
        im = raw.raw_image_visible.astype(np.float32)
        raw_pattern = raw.raw_pattern
        R = np.where(raw_pattern==0)
        G1 = np.where(raw_pattern==1)
        B = np.where(raw_pattern==2)
        G2 = np.where(raw_pattern==3)
        
        white_point = 16383
        img_shape = im.shape
        H = img_shape[0]
        W = img_shape[1]

        out = np.stack((im[R[0][0]:H:2,R[1][0]:W:2], #RGBG
                        im[G1[0][0]:H:2,G1[1][0]:W:2],
                        im[B[0][0]:H:2,B[1][0]:W:2],
                        im[G2[0][0]:H:2,G2[1][0]:W:2]), axis=0).astype(np.float32)

        black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

        out = (out - black_level) / (white_point - black_level)
        out = np.clip(out, 0, 1)
    
        return out

if __name__ == '__main__':
    ELD_SonyA7S2_dataset = ELDDataset(data_dir='E:\Deep Learning\datasets\RAW\ELD', 
                                    camera='SonyA7S2', pairs_file_path='SonyA7S2_100.txt', split='test', patch_size=512)
    data = ELD_SonyA7S2_dataset[7]
    print(len(ELD_SonyA7S2_dataset))
    input, gt, input_path, gt_path, meta = data['input_raw'], data['gt_raw'], data['input_path'], data['gt_path'], data['input_metainfo']
    print(input_path, gt_path)
    print(type(input), type(gt))
    print(input.shape, gt.shape)
    print(input.min(), input.max(), gt.min(), gt.max())
    print(meta)

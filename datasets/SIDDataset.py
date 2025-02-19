import cv2
import exifread
import numpy as np
import os
import re
import rawpy
import torch
from torch.utils import data
import sys
sys.path.append('/root/autodl-tmp/Generalization')
sys.path.append('E:\Deep Learning\MetaRawFormer\MetaRawFormer')

from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class SIDSonyDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, metarange_file='metarange.txt', patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        metarange_file = os.path.join(data_dir, metarange_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        assert os.path.exists(metarange_file), "metarange_file: {} not found.".format(metarange_file)
        self.image_list_file = image_list_file
        self.metarange_file = metarange_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 512
        self.white_level = 16383
        
        #===================================================================
        #self.keys_min_max=[[1/3200, 30],[50, 25600],[16/5, 22],[21, 223]]
        self.keys_min_max = {}
        with open(self.metarange_file, 'r') as f:
            for i, key_min_max in enumerate(f):
                key_min_max = key_min_max.strip()
                key, _min, _max = key_min_max.split(' ')
                self.keys_min_max[key] = [eval(_min), eval(_max)]
        #===================================================================
        
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW
                input_path, gt_path = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                _id = os.path.basename(input_path)#10003_00_10s.ARW
                _id, extension = os.path.splitext(_id)#10003_00_10s    .ARW
                metainfo_path = os.path.join(self.data_dir, 'Sony/meta', _id + '.txt')
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
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio),
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_dir, 'Sony', input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_dir, 'Sony', gt_path))
        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy

        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)
            
        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch = self.crop_random_patch(input_raw, gt_raw, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        input_metainfo = self.getMetaInfoTensor(info['meta'])
        #input_metainfo = self.getMetaInfoTensor(self.metainfo(os.path.join(self.data_dir, 'Sony', input_path)))
        #gt_metainfo = self.getMetaInfoTensor(self.metainfo(os.path.join(self.data_dir, 'Sony', gt_path)))

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'input_metainfo': input_metainfo,
            #'gt_metainfo': gt_metainfo,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def getMetaInfoTensor(self, metainfo):
        res = []
        for i, (k, v) in enumerate(metainfo.items()):
            min = self.keys_min_max[k][0]
            max = self.keys_min_max[k][1]
            res.append(torch.tensor((v - min) / (max - min), dtype=torch.float32))
        return torch.tensor(res)

    def pack_raw(self, raw):
        # pack Bayer image to 4 channels (RGBG)
        im = raw.raw_image_visible.astype(np.uint16)
        H, W = im.shape
        im = np.expand_dims(im, axis=0)
        out = np.concatenate((im[:, 0:H:2, 0:W:2],
                            im[:, 0:H:2, 1:W:2],
                            im[:, 1:H:2, 1:W:2],
                            im[:, 1:H:2, 0:W:2]), axis=0)
        return out

    def crop_random_patch(self, input_raw, gt_raw, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]

        return input_raw, gt_raw

    def metainfo(self, rawpath):
        with open(rawpath, 'rb') as f:
            tags = exifread.process_file(f)
            _, suffix = os.path.splitext(os.path.basename(rawpath))
            tag_prefix = 'Image' if suffix == '.dng' else 'EXIF'
            info = {
                'ExposureTime': eval(str(tags[f'{tag_prefix} ExposureTime'])),
                'ISOSpeedRating': eval(str(tags[f'{tag_prefix} ISOSpeedRatings'])),
                'FNumber': eval(str(tags[f'{tag_prefix} FNumber'])),
                'FocalLength': eval(str(tags[f'{tag_prefix} FocalLength']))
            }
        return info


@DATASET_REGISTRY.register()
class SIDFujiDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, metarange_file='metarange.txt', patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        metarange_file = os.path.join(data_dir, metarange_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        assert os.path.exists(metarange_file), "metarange_file: {} not found.".format(metarange_file)
        self.image_list_file = image_list_file
        self.metarange_file = metarange_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 1024
        self.white_level = 16383
        
        #===================================================================
        self.keys_min_max = {}
        with open(self.metarange_file, 'r') as f:
            for i, key_min_max in enumerate(f):
                key_min_max = key_min_max.strip()
                key, _min, _max = key_min_max.split(' ')
                self.keys_min_max[key] = [eval(_min), eval(_max)]
        #===================================================================
        
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW
                input_path, gt_path = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                _id = os.path.basename(input_path)#10003_00_10s.ARW
                _id, extension = os.path.splitext(_id)#10003_00_10s    .ARW
                metainfo_path = os.path.join(self.data_dir, 'Fuji/meta', _id + '.txt')
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
                    'input_exposure': input_exposure,
                    'gt_exposure': gt_exposure,
                    'ratio': np.float32(ratio),
                })
        print("processing: {} images for {}".format(len(self.img_info), self.split))

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']#./short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_dir, 'Fuji', input_path))
        input_raw = self.pack_raw(input_raw)# numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_dir, 'Fuji', gt_path))
        gt_raw = self.pack_raw(gt_raw)#numpy

        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)

        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch = self.crop_random_patch(input_raw, gt_raw, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        input_metainfo = self.getMetaInfoTensor(info['meta'])
        
        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'input_metainfo': input_metainfo,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def getMetaInfoTensor(self, metainfo):
        res = []
        for i, (k, v) in enumerate(metainfo.items()):
            min = self.keys_min_max[k][0]
            max = self.keys_min_max[k][1]
            res.append(torch.tensor((v - min) / (max - min), dtype=torch.float32))
        return torch.tensor(res)
    
    def pack_raw(self, raw):
        # pack XTrans image to 9 channels ()
        im = raw.raw_image_visible.astype(np.uint16)

        H, W = im.shape
        h1 = 0
        h2 = H // 6 * 6
        w1 = 0
        w2 = W // 6 * 6
        out = np.zeros((9, h2 // 3, w2 // 3), dtype=np.uint16)
        
        # 0 R
        out[0, 0::2, 0::2] = im[h1:h2:6, w1:w2:6]
        out[0, 0::2, 1::2] = im[h1:h2:6, w1+4:w2:6]
        out[0, 1::2, 0::2] = im[h1+3:h2:6, w1+1:w2:6]
        out[0, 1::2, 1::2] = im[h1+3:h2:6, w1+3:w2:6]

        # 1 G
        out[1, 0::2, 0::2] = im[h1:h2:6, w1+2:w2:6]
        out[1, 0::2, 1::2] = im[h1:h2:6, w1+5:w2:6]
        out[1, 1::2, 0::2] = im[h1+3:h2:6, w1+2:w2:6]
        out[1, 1::2, 1::2] = im[h1+3:h2:6, w1+5:w2:6]

        # 1 B
        out[2, 0::2, 0::2] = im[h1:h2:6, w1+1:w2:6]
        out[2, 0::2, 1::2] = im[h1:h2:6, w1+3:w2:6]
        out[2, 1::2, 0::2] = im[h1+3:h2:6, w1:w2:6]
        out[2, 1::2, 1::2] = im[h1+3:h2:6, w1+4:w2:6]

        # 4 R
        out[3, 0::2, 0::2] = im[h1+1:h2:6, w1+2:w2:6]
        out[3, 0::2, 1::2] = im[h1+2:h2:6, w1+5:w2:6]
        out[3, 1::2, 0::2] = im[h1+5:h2:6, w1+2:w2:6]
        out[3, 1::2, 1::2] = im[h1+4:h2:6, w1+5:w2:6]

        # 5 B
        out[4, 0::2, 0::2] = im[h1+2:h2:6, w1+2:w2:6]
        out[4, 0::2, 1::2] = im[h1+1:h2:6, w1+5:w2:6]
        out[4, 1::2, 0::2] = im[h1+4:h2:6, w1+2:w2:6]
        out[4, 1::2, 1::2] = im[h1+5:h2:6, w1+5:w2:6]

        out[5, :, :] = im[h1+1:h2:3, w1:w2:3]
        out[6, :, :] = im[h1+1:h2:3, w1+1:w2:3]
        out[7, :, :] = im[h1+2:h2:3, w1:w2:3]
        out[8, :, :] = im[h1+2:h2:3, w1+1:w2:3]
        return out

    def crop_random_patch(self, input_raw, gt_raw, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (9,1344,2010)
        '''
        _, H, W = input_raw.shape
        yy, xx = np.random.randint(0, H - patch_size),  np.random.randint(0, W - patch_size)
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]

        return input_raw, gt_raw


        info = self.img_info[index]
        input_path = info['input_path']#./short/10003_00_0.04s.ARW 
        input_raw = rawpy.imread(os.path.join(self.data_dir, 'Sony', input_path))
        input_raw = self.pack_raw(input_raw)#(4,2848/2,4256/2) numpy

        gt_path = info['gt_path']
        gt_raw = rawpy.imread(os.path.join(self.data_dir, 'Sony', gt_path))
        gt_raw = self.pack_raw(gt_raw)#(4,2848/2,4256/2) numpy

        input_raw = (np.float32(input_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        gt_raw = (np.float32(gt_raw) - self.black_level) / np.float32(self.white_level - self.black_level)
        if self.ratio:
            input_raw = input_raw * info['ratio']
        input_raw = np.maximum(np.minimum(input_raw, 1.0), 0.0)
        gt_raw = np.maximum(np.minimum(gt_raw, 1.0), 0.0)
            
        if self.split == 'train':
            if self.h_flip and np.random.randint(0,2) == 1:  # random horizontal flip
                input_raw = np.flip(input_raw, axis=2)
                gt_raw = np.flip(gt_raw, axis=2)
            if self.v_flip and np.random.randint(0,2) == 1:  # random vertical flip
                input_raw = np.flip(input_raw, axis=1)
                gt_raw = np.flip(gt_raw, axis=1)
            if self.transpose and np.random.randint(0,2) == 1:  # random transpose
                input_raw = np.transpose(input_raw, (0, 2, 1))
                gt_raw = np.transpose(gt_raw, (0, 2, 1)) 
            if self.patch_size:
                input_patch, gt_raw_patch = self.crop_random_patch(input_raw, gt_raw, self.patch_size)
                input_raw = input_patch.copy()
                gt_raw = gt_raw_patch.copy()
        
        input_raw = np.ascontiguousarray(input_raw)
        gt_raw = np.ascontiguousarray(gt_raw)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        
        input_metainfo_dict = self.metainfo(os.path.join(self.data_dir, 'Sony', input_path))
        input_metainfo = self.getMetaInfoIdx(input_metainfo_dict)
        gt_metainfo = self.getMetaInfoIdx(self.metainfo(os.path.join(self.data_dir, 'Sony', gt_path)))


        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'input_metainfo_dict': input_metainfo_dict,
            'input_metainfo': input_metainfo,
            'gt_metainfo': gt_metainfo,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

if __name__=='__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    #dataset = SIDSonyDataset(data_dir='/root/autodl-tmp/datasets/SID/Sony/', image_list_file='test_00.txt',
                            #patch_size=512, split='test')
    #dataset = SIDSonyDataset(data_dir='E:\Deep Learning\datasets\RAW\SID\Sony/', image_list_file='test_00.txt', split='test')
    dataset = SIDFujiDataset(data_dir='/data/dataset/Carlos/SID/Fuji/', image_list_file='test_00.txt', split='test')
    data = dataset[7]
    input_path, input_raw, gt_raw, input_metainfo = data['input_path'], data['input_raw'], data['gt_raw'], data['input_metainfo']
    print(input_path)
    print(input_metainfo.shape, input_raw.shape, gt_raw.shape)
    print(input_raw.min(), input_raw.max(), gt_raw.min(), gt_raw.max())
    print(input_metainfo)
    exit(0)


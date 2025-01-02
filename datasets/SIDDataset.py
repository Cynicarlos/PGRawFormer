import cv2
import exifread
import numpy as np
import os
import random
import rawpy
import torch
from torch.utils import data
import sys
sys.path.append('/root/autodl-tmp/Generalization')
sys.path.append('E:\Deep Learning\MetaRawFormer\MetaRawFormer')

from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class SIDSonyDataset(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 512
        self.white_level = 16383
        
        #===================================================================
        self.keys_min_max=[[1/3200, 30],[50, 25600],[16/5, 22],[21, 223]]
        #===================================================================
        
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW
                input_path, gt_path = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
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
        mask = self.getMask(input_raw)

        input_raw = torch.from_numpy(input_raw).float()
        gt_raw = torch.from_numpy(gt_raw).float()
        input_metainfo = self.getMetaInfoTensor(self.metainfo(os.path.join(self.data_dir, 'Sony', input_path)))
        gt_metainfo = self.getMetaInfoTensor(self.metainfo(os.path.join(self.data_dir, 'Sony', gt_path)))

        return {
            'input_raw': input_raw,
            'gt_raw': gt_raw,
            'input_metainfo': input_metainfo,
            'gt_metainfo': gt_metainfo,
            'mask': mask,
            'input_path': input_path,
            'gt_path': gt_path,
            'input_exposure': info['input_exposure'],
            'gt_exposure': info['gt_exposure'],
            'ratio': info['ratio']
        }

    def getMask(self, x):
        '''
        x: numpy (4, h, w) RGBG
        '''
        light = np.transpose(x, (1, 2, 0)) * 255 #(h,w,4)
        channels = [light[:, :, i] for i in range(4)]#4 (h,w,1)
        channels = [cv2.blur(channel.astype(np.uint8), (5, 5)) for channel in channels]
        light = np.stack(channels, axis=-1) #(h,w,4)
        light = light * 1.0 / 255.0
        light = np.transpose(light, (2, 0, 1))#(4,h,w) 均值滤波后的先验去噪图像
        
        dark = x[0:1, :, :] * 0.298 + x[1:2, :, :] * 0.294 + x[2:3, :, :] * 0.114 + x[3:4, :, :] * 0.294
        light = light[0:1, :, :] * 0.298 + light[1:2, :, :] * 0.294 + light[2:3, :, :] * 0.114 + light[3:4, :, :] * 0.294
        
        dark = torch.from_numpy(dark).float()
        light = torch.from_numpy(light).float()
        diff = torch.abs(dark - light)
        #mask = torch.div(light, noise + 0.0001) #(4,h,w) noise + -> mask -

        diff_max = torch.max(diff)

        mask = diff / (diff_max + 0.0001)
        mask = torch.clamp(mask, min=0, max=1.0).float()
        return mask

    
    def getMetaInfoTensor(self, metainfo):
        infos = []
        for i, (k, v) in enumerate(metainfo.items()):
            min = self.keys_min_max[i][0]
            max = self.keys_min_max[i][1]
            infos.append(torch.tensor((v - min) / (max - min), dtype=torch.float32))
        return torch.tensor(infos)

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

    def crop_center_patch(self, input_raw, gt_raw, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        '''
        _, H, W = input_raw.shape
        yy, xx = (H - patch_size) // 2,  (W - patch_size) // 2
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]

        return input_raw, gt_raw

class SIDSonyDatasetIdx(data.Dataset):
    def __init__(self, data_dir, image_list_file, patch_size=None, split='train', transpose=False,
                h_flip=False, v_flip=False, ratio=True, **kwargs):
        assert os.path.exists(data_dir), "data_path: {} not found.".format(data_dir)
        self.data_dir = data_dir
        image_list_file = os.path.join(data_dir, image_list_file)
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self.patch_size = patch_size
        self.split = split
        self.transpose = transpose
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.ratio = ratio
        self.black_level = 512
        self.white_level = 16383
        
        #===================================================================
        self.keys = ['ExposureTime', 'ISOSpeedRating', 'FNumber', 'FocalLength']
        self.ExposureTime_table = [
            1/3200, 1/2000, 1/1600, 1/1000, 1/800, 1/500, 1/400, 1/250, 1/200, 1/160, 1/100, 1/80, 1/50,
            1/40, 1/30, 1/25, 1/20, 1/15, 1/10, 1/8, 1/5, 1/4, 2/5, 1/2, 2, 16/5, 4, 10, 30
        ]#28 intervals
        self.exposure_to_index = {i: idx for idx, i in enumerate(range(len(self.ExposureTime_table) - 1))}
    
        self.ISOSpeedRating_table=[
            50, 64, 80, 100, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600,
            2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 25600
        ]#25 intervals
        self.iso_to_index = {i: idx for idx, i in enumerate(range(len(self.ISOSpeedRating_table) - 1))}

        self.Fnumber_table = [i for i in range(1, 26)]#25 intervals
        self.fnumber_to_index = {i: idx for idx, i in enumerate(range(len(self.Fnumber_table) - 1))}
        
        self.FocalLength_table = [i for i in range(1, 252)]#250
        self.focallength_to_index = {i: idx for idx, i in enumerate(range(len(self.FocalLength_table) - 1))}
        self.dicts = {
            'ExposureTime': self.ExposureTime_table,
            'ISOSpeedRating': self.ISOSpeedRating_table,
            'FNumber': self.Fnumber_table,
            'FocalLength': self.FocalLength_table,
        }
        #===================================================================
        
        self.img_info = []
        with open(self.image_list_file, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()  # ./Sony/short/10003_00_0.04s.ARW ./Sony/long/10003_00_10s.ARW
                input_path, gt_path = img_pair.split(' ')
                input_exposure = float(os.path.split(input_path)[-1][9:-5]) # 0.04
                gt_exposure = float(os.path.split(gt_path)[-1][9:-5]) # 10
                ratio = min(gt_exposure/input_exposure, 300)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path,
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

    
    def get_exposure_index(self, exposure):
        num_intervals = len(self.ExposureTime_table)
        for idx, boundary in enumerate(self.ExposureTime_table):
            if exposure < boundary:
                return self.exposure_to_index.get(idx - 1, num_intervals - 1)
        return num_intervals - 1  # 对于超出最大范围的曝光时长，返回最后一个区间的索引

    def get_iso_index(self, iso):
        num_intervals = len(self.ISOSpeedRating_table)
        for idx, boundary in enumerate(self.ISOSpeedRating_table):
            if iso <= boundary:
                return self.iso_to_index.get(idx, num_intervals - 1)
        return num_intervals - 1  # 对于超出最大范围的ISO值，返回最后一个区间的索引
    
    def get_fnumber_index(self, iso):
        num_intervals = len(self.Fnumber_table)
        for idx, boundary in enumerate(self.Fnumber_table):
            if iso <= boundary:
                return self.fnumber_to_index.get(idx, num_intervals - 1)
        return num_intervals - 1 
    
    def get_focallength_index(self, iso):
        num_intervals = len(self.FocalLength_table)
        for idx, boundary in enumerate(self.FocalLength_table):
            if iso <= boundary:
                return self.focallength_to_index.get(idx, num_intervals - 1)
        return num_intervals - 1 

    def getMetaInfoIdx(self, metainfo):
        idx_list  = []
        #['ExposureTime', 'FNumber', 'FocalLength', 'ISOSpeedRating']
        idx_list.append(self.get_exposure_index(metainfo['ExposureTime']))
        idx_list.append(self.get_iso_index(metainfo['ISOSpeedRating']))
        idx_list.append(self.get_fnumber_index(metainfo['FNumber']))
        idx_list.append(self.get_focallength_index(metainfo['FocalLength']))

        return torch.tensor(idx_list, dtype=torch.long)

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

    def crop_center_patch(self, input_raw, gt_raw, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        '''
        _, H, W = input_raw.shape
        yy, xx = (H - patch_size) // 2,  (W - patch_size) // 2
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]

        return input_raw, gt_raw


if __name__=='__main__':
    seed = 3407
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    #dataset = SIDSonyDataset(data_dir='/root/autodl-tmp/datasets/SID/Sony/', image_list_file='test_00.txt',
                            #patch_size=512, split='test')
    dataset = SIDSonyDataset(data_dir='E:\Deep Learning\datasets\RAW\SID\Sony/', image_list_file='test_00.txt', split='test')
    data = dataset[7]
    input_raw, gt_raw, input_metainfo, gt_metainfo, mask = \
        data['input_raw'], data['gt_raw'], data['input_metainfo'], data['gt_metainfo'], data['mask']
    print(type(input_metainfo), type(gt_metainfo), type(mask))
    print(input_metainfo.shape, gt_metainfo.shape, mask.shape)
    print(input_metainfo, gt_metainfo, mask)
    print(mask.min(), mask.max())
    #exit(0)
    import matplotlib.pyplot as plt
    mask = mask.view(-1).numpy()
    #mask = [level for level in mask if level <= 0.002]
    plt.figure(figsize=(8, 6))
    plt.hist(mask, bins=50, color='blue', alpha=0.7)
    plt.title(f'mask distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()



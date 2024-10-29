import os
import torch
import rawpy
import numpy as np
import exifread
from torch.utils.data import Dataset
import sys
sys.path.append('/root/autodl-tmp/Generalization')
#Notice: The ELD dataset is only for evaluation

from utils.registry import DATASET_REGISTRY
@DATASET_REGISTRY.register()
class ELDDataset(Dataset):
    def __init__(self, datadir, camera, pairs_file_path, patch_size=None, **kwargs):
        super(ELDDataset, self).__init__()
        assert camera in ['CanonEOS70D', 'CanonEOS700D', 'NikonD850', 'SonyA7S2']
        self.datadir = datadir
        self.camera = camera
        self.patch_size = patch_size
        self.pairs_file_path=os.path.join(datadir, pairs_file_path)
                #===================================================================
        self.keys = ['ExposureTime', 'FNumber', 'FocalLength', 'ISOSpeedRating', 'MeteringMode']
        self.ExposureTime_table = [
            0, 1/3200, 1/2000, 1/1600, 1/1000, 1/800, 1/500,
            1/400, 1/250, 1/200, 1/160, 1/100, 1/80, 1/50,
            1/40, 1/30, 1/25, 1/20, 1/15, 1/10, 1/8, 1/5,
            1/4, 2/5, 1/2, 2, 16/5, 4, 10, 30
        ]#29 intervals + 1 out of table
        self.exposure_to_index = {i: idx for idx, i in enumerate(range(len(self.ExposureTime_table) - 1))}
        
        self.ISOSpeedRating_table=[
            50, 64, 80, 100, 160, 200, 250, 320, 400, 500, 
            640, 800, 1000, 1250, 1600, 2000, 2500, 3200, 
            4000, 5000, 6400, 8000, 10000, 12800, 16000, 25600
        ]#25 intervals + 1 out of table
        self.iso_to_index = {i: idx for idx, i in enumerate(range(len(self.ISOSpeedRating_table) - 1))}
        
        self.Fnumber_table = [None] + [i for i in range(25)]#26
        self.FocalLength_table = [None] + [i for i in range(1, 249)]#250
        self.MeteringMode_table = [None, 'CenterWeightedAverage', 'Pattern']#3
        self.dicts = {
            'ExposureTime': self.ExposureTime_table,
            'FNumber': self.Fnumber_table,
            'FocalLength': self.FocalLength_table,
            'ISOSpeedRating': self.ISOSpeedRating_table,
            'MeteringMode': self.MeteringMode_table
        } 
        self.img_info=[]
        with open(self.pairs_file_path, 'r') as f:
            for i, img_pair in enumerate(f):
                img_pair = img_pair.strip()
                input_path, gt_path, _ = img_pair.split(' ')
                input_path = os.path.join(datadir, self.camera, input_path)
                gt_path = os.path.join(datadir, self.camera, gt_path)
                self.img_info.append({
                    'input_path': input_path,
                    'gt_path': gt_path
                })

    def __getitem__(self, index):
        info = self.img_info[index]
        input_path = info['input_path']
        gt_path = info['gt_path']

        gt_meta = self.metainfo(gt_path)
        iso, expo = gt_meta['ISOSpeedRating'], gt_meta['ExposureTime']
        gt_expo = iso * expo
        
        input_meta = self.metainfo(input_path)
        iso, expo = input_meta['ISOSpeedRating'], input_meta['ExposureTime']

        ratio = gt_expo / (iso * expo)
        
        with rawpy.imread(input_path) as raw:
            input = self.pack_raw_bayer(raw) * ratio

        with rawpy.imread(gt_path) as raw:
            gt = self.pack_raw_bayer(raw)
            
        if self.patch_size:
            input, gt = self.crop_center_patch(input, gt, self.patch_size)

        input = np.maximum(np.minimum(input, 1.0), 0)
        gt = np.maximum(np.minimum(gt, 1.0), 0)
        
        input = np.ascontiguousarray(input)
        gt = np.ascontiguousarray(gt)
        input = torch.from_numpy(input)
        gt = torch.from_numpy(gt)
        
        input_metainfoidx = self.getMetaInfoIdx(input_meta)
        gt_metainfoidx = self.getMetaInfoIdx(gt_meta)

        data = {
            'input_raw': input, 
            'gt_raw': gt, 
            'input_metainfoidx': input_metainfoidx,
            'gt_metainfoidx': gt_metainfoidx,
            'input_path':input_path, 
            'gt_path': gt_path
        }
        
        return data

    def __len__(self):
        return len(self.img_info)
    
    def metainfo(self, rawpath):
        with open(rawpath, 'rb') as f:
            tags = exifread.process_file(f)
            _, suffix = os.path.splitext(os.path.basename(rawpath))
            tag_prefix = 'Image' if suffix == '.dng' else 'EXIF'
            info = {
                'ExposureTime': eval(str(tags[f'{tag_prefix} ExposureTime'])),
                'FNumber': eval(str(tags[f'{tag_prefix} FNumber'])),
                'FocalLength': eval(str(tags[f'{tag_prefix} FocalLength'])),
                'ISOSpeedRating': eval(str(tags[f'{tag_prefix} ISOSpeedRatings'])),
                'MeteringMode': str(tags[f'{tag_prefix} MeteringMode'])
            }
        return info
    
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
    
    def get_index_from_dict(self, value, key):
        if value in self.dicts[key]:
            return self.dicts[key].index(value)
        else:
            return 0
    
    def getMetaInfoIdx(self, metainfo):
        idx_list  = []
        #['ExposureTime', 'FNumber', 'FocalLength', 'ISOSpeedRating', 'MeteringMode']
        idx_list.append(self.get_exposure_index(metainfo['ExposureTime']))
        idx_list.append(self.get_iso_index(metainfo['ISOSpeedRating']))
        idx_list.append(self.get_index_from_dict(metainfo['FNumber'], 'FNumber'))
        idx_list.append(self.get_index_from_dict(metainfo['FocalLength'], 'FocalLength'))
        idx_list.append(self.get_index_from_dict(metainfo['MeteringMode'], 'MeteringMode'))
            
        return torch.tensor(idx_list , dtype=torch.long)
    
    def crop_center_patch(self, input_raw, gt_raw, patch_size):
        '''
        input_raw, gt_raw: numpy with shape (4,H/2,W/2)
        '''
        _, H, W = input_raw.shape
        yy, xx = (H - patch_size) // 2,  (W - patch_size) // 2
        input_raw = input_raw[:, yy:yy + patch_size, xx:xx + patch_size]
        gt_raw = gt_raw[:, yy:yy + patch_size, xx:xx + patch_size]
    
        return input_raw, gt_raw
    
    '''
    def metainfo(self, rawpath):
        with open(rawpath, 'rb') as f:
            tags = exifread.process_file(f)
            _, suffix = os.path.splitext(os.path.basename(rawpath))
            if suffix == '.dng':
                expo = eval(str(tags['Image ExposureTime']))
                iso = eval(str(tags['Image ISOSpeedRatings']))
            else:
                expo = eval(str(tags['EXIF ExposureTime']))
                iso = eval(str(tags['EXIF ISOSpeedRatings']))
        return iso, expo
    '''



    
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
    ELD_SonyA7S2_dataset = ELDDataset(datadir='E:\Deep Learning\datasets\RAW\ELD', 
                                      camera='SonyA7S2',pairs_file_path='SonyA7S2_100.txt',patch_size=512)
    data = ELD_SonyA7S2_dataset[7]
    print(len(ELD_SonyA7S2_dataset))
    input, gt, input_path, gt_path = data['input'], data['gt'], data['input_path'], data['gt_path']
    print(input_path, gt_path)
    print(type(input), type(gt))
    print(input.shape, gt.shape)
    print(input.min(), input.max(), gt.min(), gt.max())

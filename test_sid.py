import argparse
import os
import torch
from datasets.SIDDataset import SIDSonyDataset
from tqdm import tqdm
import yaml
from models import build_model
from tqdm import tqdm
from utils import set_random_seed
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.data import crop_tow_patch, crop_four_patch, auto_crop, auto_reconstruct


@torch.no_grad()
def test(model, dataloader, ratio, merge_test=False, num_patch=None, with_metainfo=False):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    samples = len(dataloader.dataset)
    tqdm_loader = tqdm(dataloader, desc=f"Evaluating SID_Sony ratio {ratio}", leave=False)
    
    with open(f'results/SID/Sony_Ratio_{ratio}.txt', 'w') as f:
        for idx, data in enumerate(tqdm_loader):
            input = data['input_raw'].cuda()
            gt = data['gt_raw'].cuda()
            input_path = data['input_path'][0]
            gt_path = data['gt_path'][0]
            if with_metainfo:
                input_metainfoidx = data['input_metainfo'].cuda()
                
            if merge_test:
                assert num_patch is not None
                patch_size = config['test'].get('patch_size', None)
                if num_patch == -1:
                    assert patch_size is not None
                    inputs = auto_crop(input)
                    gts = auto_crop(gt)
                    if with_metainfo:
                        preds = [model(input_patch, input_metainfoidx) for input_patch in inputs]
                    else:
                        preds = [model(input_patch) for input_patch in inputs]
                    #h, w = input.shape[2:]
                    #num_row_patch, num_col_patch = w // patch_size, h // patch_size
                    #pred = auto_reconstruct(pres, num_row_patch=num_row_patch, num_col_patch=num_col_patch, patch_size=patch_size)

                elif num_patch == 2:
                    inputs = crop_tow_patch(input)
                    gts = crop_tow_patch(gt)
                    if with_metainfo:
                        preds = [model(patch, input_metainfoidx) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]
                    #pred = torch.cat(preds, dim=3)

                elif num_patch == 4:
                    inputs = crop_four_patch(input)
                    gts = crop_four_patch(gt)
                    if with_metainfo:
                        preds = [model(patch, input_metainfoidx) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]
                    #x_top = torch.cat(preds[:2], dim=3)
                    #x_bottom = torch.cat(preds[2:], dim=3)
                    #pred = torch.cat([x_top, x_bottom], dim=2)
                preds = [torch.clamp(pred, 0, 1) for pred in preds]
                psnrs = [get_psnr_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                ssims = [get_ssim_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                psnr = sum(psnrs) / len(psnrs)
                ssim = sum(ssims) / len(ssims)
            else:
                if with_metainfo:
                    pred = model(input, input_metainfoidx)
                else:
                    pred = model(input)
                pred = torch.clamp(pred, 0, 1)
                psnr = get_psnr_torch(pred, gt, data_range=1.0)
                ssim = get_ssim_torch(pred, gt, data_range=1.0)

            f.write(f"input:{input_path}    gt:{gt_path}    psnr:{psnr.item():.4f}   ssim:{ssim.item():.4f}\n")

            psnr_sum += psnr.item()
            ssim_sum += ssim.item()

            tqdm_loader.set_postfix({'psnr':f'{psnr.item():.4f}','ssim':f'{ssim.item():.4f}', 'avg_psnr': f'{psnr_sum/(idx+1):.4f}', 'avg_ssim':f'{ssim_sum/(idx+1):.4f}'}, refresh=True)

        average_psnr = psnr_sum / samples
        average_ssim = ssim_sum / samples

        f.write(f'psnr:{average_psnr:.4f}       ssim:{average_ssim:.4f}')
    print(f'Ratio_{ratio}   psnr:{average_psnr:.4f}     ssim:{average_ssim:.4f}')
    return psnr_sum, ssim_sum, samples

if __name__ == "__main__":
    os.makedirs('results/SID', exist_ok=True)
    data_dir='/root/autodl-tmp/datasets/SID/Sony'
    #data_dir='/data/dataset/Carlos/SID/Sony' #30
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metainfo', action='store_true', default=False)
    parser.add_argument('--merge_test', action='store_true', default=False)
    #parser.add_argument('--num_patch', type=int, default=None)
    args = parser.parse_args()
    
    with open('configs/sony.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    model_name, model = build_model(config['model'])
    model = model.cuda()
    checkpoint = torch.load('/root/autodl-tmp/MetaRawDenoising/runs/SONY/checkpoints/best_model.pth') 
    #checkpoint = torch.load('/data/model/Carlos/RAWDenoising/runs/SONY/checkpoints/best_model.pth') #30
    model.load_state_dict(checkpoint['model'])

    test_files = ['test_00_100.txt', 'test_00_250.txt', 'test_00_300.txt']
    ratios = [100, 250, 300]
    
    all_ratio_psnr_sum = 0
    all_ratio_ssim_sum = 0
    total_samples = 0
    
    for test_file, ratio in zip(test_files, ratios):
        dataset = SIDSonyDataset(data_dir=data_dir, image_list_file=test_file, split='test')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
        psnr, ssim, samples = test(model, dataloader, ratio=ratio, 
                                    merge_test=args.merge_test, num_patch=2, with_metainfo=args.with_metainfo)
        all_ratio_psnr_sum += psnr
        all_ratio_ssim_sum += ssim
        total_samples += samples
    all_ratio_avg_psnr, all_ratio_avg_ssim = all_ratio_psnr_sum / total_samples, all_ratio_ssim_sum / total_samples
    print(f'total samples:{total_samples}    all_ratio_psnr:{all_ratio_avg_psnr}    all_ratio_ssim:{all_ratio_avg_ssim}')
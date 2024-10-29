import argparse
import os
import torch
from datasets.ELDDataset import ELDDataset
from tqdm import tqdm
import yaml
from models import build_model
from tqdm import tqdm
from utils import set_random_seed
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.data import crop_tow_patch, crop_four_patch, auto_crop, auto_reconstruct

@torch.no_grad()
def test(model, dataloader, camera, ratio, merge_test=False, num_patch=None, with_metainfo=False):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    total_samples = len(dataloader.dataset)
    tqdm_loader = tqdm(dataloader, desc=f"Evaluating Camara {camera}:", leave=False)
    
    with open(f'results/ELD/{camera}_{ratio}.txt', 'w') as f:
        for idx, data in enumerate(tqdm_loader):
            input = data['input_raw'].cuda()
            gt = data['gt_raw'].cuda()
            input_path = data['input_path'][0]
            gt_path = data['gt_path'][0]
            if with_metainfo:
                input_metainfoidx = data['input_metainfoidx'].cuda()
            if merge_test:
                assert num_patch in [-1, 2, 4]
                if num_patch == -1:
                    patch_size = config['test'].get('patch_size', None)
                    assert patch_size is not None
                    inputs = auto_crop(input)
                    if with_metainfo:
                        pres = [model(input_patch, input_metainfoidx) for input_patch in inputs]
                    else:
                        pres = [model(input_patch) for input_patch in inputs]
                    h, w = input.shape[2:]
                    num_row_patch, num_col_patch = w // patch_size, h // patch_size
                    pred = auto_reconstruct(pres, num_row_patch=num_row_patch, num_col_patch=num_col_patch, patch_size=patch_size)

                elif num_patch == 2:
                    inputs = crop_tow_patch(input)
                    if with_metainfo:
                        preds = [model(patch, input_metainfoidx) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]
                    pred = torch.cat(preds, dim=3)

                elif num_patch == 4:
                    inputs = crop_four_patch(input)
                    if with_metainfo:
                        preds = [model(patch, input_metainfoidx) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]
                    x_top = torch.cat(preds[:2], dim=3)
                    x_bottom = torch.cat(preds[2:], dim=3)
                    pred = torch.cat([x_top, x_bottom], dim=2)
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
        
        average_psnr = psnr_sum / total_samples
        average_ssim = ssim_sum / total_samples

        f.write(f'psnr:{average_psnr:.4f}       ssim:{average_ssim:.4f}')
    print(f'Camera {camera}    Ratio {ratio}    PSNR:{average_psnr:.4f}    SSIM:{average_ssim:.4f}')
    return psnr_sum, ssim_sum, total_samples

if __name__ == "__main__":
    os.makedirs('results/ELD', exist_ok=True)
    #datadir='/root/autodl-tmp/datasets/ELD/'
    datadir='/data/dataset/Carlos/ELD' #30
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metainfo', action='store_true', default=False)
    args = parser.parse_args()
    
    with open('configs/sony.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    model_name, model = build_model(config['model'])
    model = model.cuda()
    #checkpoint = torch.load('/root/autodl-tmp/Generalization/runs/SONY/checkpoints/best_model.pth')
    checkpoint = torch.load('/data/model/Carlos/RAWDenoising/runs/SONY/checkpoints/best_model.pth') #30
    model.load_state_dict(checkpoint['model'])

    cameras = ['SonyA7S2', 'NikonD850', 'CanonEOS70D', 'CanonEOS700D']
    #cameras = ['NikonD850', 'CanonEOS70D', 'CanonEOS700D']
    #ratios = [1, 10, 100, 200]
    ratios = [100, 200]
    for camera in cameras:
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0
        for ratio in ratios:
            pairs_file_path = os.path.join(datadir, f'{camera}_{ratio}.txt')
            dataset = ELDDataset(datadir=datadir, camera=camera, pairs_file_path=pairs_file_path,patch_size=None)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
            psrn, ssim, samples = test(model, dataloader, camera=camera, ratio=ratio, merge_test=True, num_patch=2, with_metainfo=args.with_metainfo)
            total_psnr += psrn
            total_ssim += ssim
            total_samples += samples
        all_ratio_avg_psnr, all_ratio_avg_ssim = total_psnr / total_samples, total_ssim / total_samples
        print(f'total samples:{total_samples}    all_ratio_psnr:{all_ratio_avg_psnr}    all_ratio_ssim:{all_ratio_avg_ssim}')


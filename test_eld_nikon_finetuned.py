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
                input_metainfo = data['input_metainfo'].cuda()
            if merge_test:
                assert num_patch is not None
                patch_size = config['test'].get('patch_size', None)
                if num_patch == -1:
                    assert patch_size is not None
                    inputs = auto_crop(input)
                    gts = auto_crop(gt)
                    if with_metainfo:
                        preds = [model(input_patch, input_metainfo) for input_patch in inputs]
                    else:
                        preds = [model(input_patch) for input_patch in inputs]

                elif num_patch == 2:
                    inputs = crop_tow_patch(input)
                    gts = crop_tow_patch(gt)
                    if with_metainfo:
                        preds = [model(patch, input_metainfo) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]


                elif num_patch == 4:
                    inputs = crop_four_patch(input)
                    gts = crop_four_patch(gt)
                    if with_metainfo:
                        preds = [model(patch, input_metainfo) for patch in inputs]
                    else:
                        preds = [model(patch) for patch in inputs]

                preds = [torch.clamp(pred, 0, 1) for pred in preds]
                psnrs = [get_psnr_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                ssims = [get_ssim_torch(pred, gt, data_range=1.0) for (pred, gt) in zip(preds, gts)]
                psnr = sum(psnrs) / len(psnrs)
                ssim = sum(ssims) / len(ssims)
            else:
                if with_metainfo:
                    pred = model(input, input_metainfo)
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
    datadir = '/data/dataset/Carlos/ELD/' #31
    
    with open('configs/ELD_NikonD850.yaml', 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    model_name, model = build_model(config['model'])
    model = model.cuda()
    
    checkpoint = torch.load('/data/models/Carlos/RAWDenoising/runs/ELD_NikonD850/checkpoints/best_model.pth') #31

    model.load_state_dict(checkpoint['model'])
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_metainfo', action='store_true', default=False)
    parser.add_argument('--merge_test', action='store_true', default=False)
    parser.add_argument('--num_patch', type=int, default=4)
    args = parser.parse_args()

    cameras = ['NikonD850']
    ratios = [100, 200]
    for camera in cameras:
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0
        for ratio in ratios:
            pairs_file_path = os.path.join(datadir, f'{camera}_WO_Finetune_{ratio}.txt')
            dataset = ELDDataset(datadir=datadir, camera=camera, pairs_file_path=pairs_file_path,patch_size=None)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
            psrn, ssim, samples = test(model, dataloader, camera=camera, ratio=ratio, 
                                    merge_test=args.merge_test, num_patch=args.num_patch, with_metainfo=args.with_metainfo)
            total_psnr += psrn
            total_ssim += ssim
            total_samples += samples
        all_ratio_avg_psnr, all_ratio_avg_ssim = total_psnr / total_samples, total_ssim / total_samples
        print(f'total samples:{total_samples}    all_ratio_psnr:{all_ratio_avg_psnr}    all_ratio_ssim:{all_ratio_avg_ssim}')


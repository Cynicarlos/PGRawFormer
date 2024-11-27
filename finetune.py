import argparse
import os
import shutil
import time
import torch
import yaml

from datasets import build_train_loader, build_valid_loader, build_test_loader
from models import build_model
from timm.utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import format_time, get_grad_norm, load_pretrained_model, save_checkpoint, set_random_seed
from utils.logger import CustomLogger
from utils.loss import build_loss
from utils.metrics import get_psnr_torch, get_ssim_torch
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler
from utils.data import crop_tow_patch, crop_four_patch, auto_crop, auto_reconstruct


def main(config, args):
    writer = SummaryWriter(os.path.join(config['output'], 'tb_logs'))
    train_dataloader = build_train_loader(config['data'])
    valid_dataloader = build_test_loader(config['data'])
    
    logger.info(f"Creating model:{config['name']}/{config['model']['name']}")
    model_name, model = build_model(config['model'])
    model.cuda()

    optimizer = build_optimizer(config['train'], model)
    lr_scheduler = build_scheduler(config['train'], optimizer)
    critierion = build_loss(config['loss'])

    max_psnr = 0.0
    psnr = 0.0
    max_ssim = 0.0
    ssim = 0.0
    total_epochs = config['train']['epochs']

    load_pretrained_model(config, model, logger)

    logger.info("----------------------------------------Start Finetuning----------------------------------------")

    for epoch in range(1, total_epochs+1):
        epoch_start = time.time()
        train_total_loss, train_time = train_one_epoch(config, model, critierion, train_dataloader, optimizer, epoch, lr_scheduler, writer, with_metainfo=args.with_metainfo)
        logger.info(f'Train: [{epoch}/{config["train"]["epochs"]}] total loss: {train_total_loss:.6f} lr: {optimizer.param_groups[0]["lr"]:.6f} train_time: {train_time}')
        
        psnr, ssim, valid_time = validate(config, model, valid_dataloader, epoch, writer, args.with_metainfo)
        max_psnr = max(max_psnr, psnr)
        max_ssim = max(max_ssim, ssim)
        save_checkpoint(config, epoch, model, psnr, max_psnr, optimizer, lr_scheduler, is_best=(psnr == max_psnr))

        logger.info(f'Valid: [{epoch}/{config["train"]["epochs"]}]  PSNR: {psnr:.2f}  Max_PSNR: {max_psnr:.2f}  SSIM:{ssim:.4f}  Max_SSIM: {max_ssim:.4f}  valid_time: {valid_time}')
        writer.add_scalar('eval/max_psnr', max_psnr, epoch)
        writer.add_scalar('eval/max_ssim', max_ssim, epoch)

        epoch_end = time.time()
        epoch_time = format_time(epoch_end - epoch_start)
        logger.info(f'Epoch {epoch} time: {epoch_time}')

def train_one_epoch(config, model, critierion, data_loader, optimizer, epoch, lr_scheduler, writer, with_metainfo=False):
    start_time = time.time()
    model.train()
    optimizer.zero_grad()

    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    losses_meter = [AverageMeter() for _ in range(len(critierion))]


    tqdm_loader = tqdm(data_loader,desc=f"Train: [{epoch}/{config['train']['epochs']}]",leave=False)
    for idx, data in enumerate(tqdm_loader):
        input_raw = data['input_raw'].cuda(non_blocking=True)
        gt_raw = data['gt_raw'].cuda(non_blocking=True)
        if with_metainfo:
            input_metainfo = data['input_metainfo'].cuda(non_blocking=True)
            pred = model(input_raw, input_metainfo)#pred_raw
        else:
            pred = model(input_raw)#pred_raw

        losses = critierion(pred, gt_raw)
        loss = sum(losses)

        optimizer.zero_grad()
        loss.backward()

        if config['train'].get('clip_grad'):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()

        batch_size = config['data']['train']['batch_size']
        for _loss_meter, _loss in zip(losses_meter, losses):
            _loss_meter.update(_loss.item(), batch_size)    
        loss_meter.update(loss.item(), batch_size)
        norm_meter.update(grad_norm)
        tqdm_loader.set_postfix({'l':f'{loss_meter.val:.6f}', 'a_l':f'{loss_meter.avg:.6f}', 
                                'lr':f"{optimizer.param_groups[0]['lr']:.6f}"}, refresh=True)
    
    lr_scheduler.step(epoch)
    tensor_board_dict = {'train/loss_total':loss_meter.avg}
    for index, (_loss_meter) in enumerate(losses_meter):
        tensor_board_dict[f'train/loss_{index}'] = _loss_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)

    end_time = time.time()
    train_time = format_time(end_time-start_time)

    return loss_meter.avg, train_time

@torch.no_grad()
def validate(config, model, data_loader, epoch, writer, with_metainfo=False):
    start_time = time.time()
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    tqdm_loader = tqdm(data_loader,desc=f"Valid: [{epoch}/{config['train']['epochs']}]",leave=False)
    for idx, data in enumerate(tqdm_loader):
        input_path, gt_path = data['input_path'][0], data['gt_path'][0]
        
        input_raw = data['input_raw'].cuda(non_blocking=True)
        gt_raw = data['gt_raw'].cuda(non_blocking=True)
        
        if with_metainfo:
            input_metainfo = data['input_metainfo'].cuda(non_blocking=True)
        
        if config['test']['merge_test']:
            assert config['test']['num_patch'] is not None
            patch_size = config['test'].get('patch_size', None)

            if config['test']['num_patch'] == -1:
                assert patch_size is not None
                inputs = auto_crop(input_raw)
                gts = auto_crop(gt_raw)
                if with_metainfo:
                    pres = [model(input_patch, input_metainfo) for input_patch in inputs]
                else:
                    pres = [model(input_patch) for input_patch in inputs]

            elif config['test']['num_patch'] == 2:
                inputs = crop_tow_patch(input_raw)
                gts = crop_tow_patch(gt_raw)
                if with_metainfo:
                    preds = [model(patch, input_metainfo) for patch in inputs]
                else:
                    preds = [model(patch) for patch in inputs]

            elif config['test']['num_patch'] == 4:
                inputs = crop_four_patch(input_raw)
                gts = crop_four_patch(gt_raw)
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
                pred = model(input_raw, input_metainfo)
            else:
                pred = model(input_raw)
            pred = torch.clamp(pred, 0, 1)
            psnr = get_psnr_torch(pred, gt_raw, data_range=1.0)
            ssim = get_ssim_torch(pred, gt_raw, data_range=1.0)


        batch_size = config['data']['test']['batch_size']


        psnr_meter.update(psnr.item(), batch_size)
        ssim_meter.update(ssim.item(), batch_size)

        tqdm_loader.set_postfix({'psnr':f'{psnr_meter.val:.2f}', 'a_psnr': f'{psnr_meter.avg:.2f}',
                                'ssim':f'{ssim_meter.val:.4f}', 'a_ssim': f'{ssim_meter.avg:.4f}'},
                                refresh=True)
        os.makedirs(f"{config['output']}/results", exist_ok=True)
        with open(f"{config['output']}/results/Epoch_{epoch}_results.txt", 'a') as f:
            f.write(f"input:{input_path}    gt:{gt_path}    psnr:{psnr_meter.val:.2f}   ssim:{ssim_meter.val:.4f}\n")

    tensor_board_dict = {}
    tensor_board_dict['eval/psnr'] = psnr_meter.avg
    tensor_board_dict['eval/ssim'] = ssim_meter.avg
    for log_key, log_value in tensor_board_dict.items():
        writer.add_scalar(log_key, log_value, epoch)
    
    end_time = time.time()
    valid_time = format_time(end_time-start_time)
    return psnr_meter.avg, ssim_meter.avg, valid_time

@torch.no_grad()
def validate_metric(pred, gt):
    pred = torch.clamp(pred, 0, 1) 
    gt = torch.clamp(gt, 0, 1)
    psnrs = get_psnr_torch(pred, gt, data_range=1.0)
    ssims = get_ssim_torch(pred, gt, data_range=1.0)
    return psnrs.mean(), ssims.mean()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', required=True, type=str, help='Path to option YAML file.')
    parser.add_argument('--with_metainfo', action='store_true', default=False)
    args = parser.parse_args()
    with open(args.cfg, 'r') as file:
        config = yaml.safe_load(file)
    set_random_seed(config['manual_seed'])
    
    config['output'] = os.path.join(config.get('output', 'runs'), config['name'])
    os.makedirs(config['output'], exist_ok=True)
    os.makedirs(os.path.join(config['output'],'checkpoints'), exist_ok=True)

    start_time = time.strftime("%y%m%d-%H%M", time.localtime())
    logger = CustomLogger(log_file_path=f"{config['output']}/training_log.txt")
    path = os.path.join(config['output'], f"{start_time}.yaml")
    shutil.copy(args.cfg, path)

    model_name = config['model']['name']
    shutil.copy(f"models/{model_name}.py", config['output'])
    current_cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
    logger.info(f"Current CUDA Device: {current_cuda_device.name}, Total Mem: {int(current_cuda_device.total_memory / 1024 / 1024)}MB")
    main(config, args)



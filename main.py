# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import os
import sys
import random
import time
from option import Options
from pathlib import Path
from typing import Optional
from util.logger import setup_logger, log_first_n
from util.collect_env import collect_env_info
from loguru import logger
from tabulate import tabulate
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
import util.misc as utils
from datasets import build_dataset
from engine import evaluate_hoi, train_one_epoch
from models import build_model
from util.slconfig import SLConfig
from copy import deepcopy
import json
import wandb


def build_model_main(args):
    model, criterion, postprocessors = build_model(args)
    
    return model, criterion, postprocessors


@logger.catch
def main(args):
    utils.init_distributed_mode(args)

    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)

    save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
    cfg.dump(save_cfg_path)
    save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
    with open(save_json_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    if args.output_dir and utils.get_rank() == 0:
        setup_logger(args.output_dir, distributed_rank=0)  # master only
        log_first_n("INFO", f"Training log of {args.arch}", key="message")
        logger.info("\n{}".format(collect_env_info()))
        logger.info("\n{}".format(tabulate([(k, v) for k,v in vars(args).items()])))
    # torch.autograd.set_detect_anomaly(True)
    
    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)

    if args.use_wandb and ((not args.distributed or (args.distributed and args.rank == 0))):
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
            )
    else:
        run = None

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    if args.output_dir and utils.get_rank() == 0:
        logger.info("\n{}".format(model))



    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if args.output_dir and utils.get_rank() == 0:
        logger.info("\nnumber of params: {}".format(n_parameters))

    param_dicts = [
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr":
            args.lr_backbone,
        },
    ]
        
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    
    if args.resume and args.pretrain_model_path:
        raise ValueError("resume and pretrain_model_path cannot be used at the same time")

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    if not args.resume and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        model_without_ddp.load_state_dict(checkpoint, strict=True)

    if args.eval:
        print("Start evaluating")
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, output_dir, -1, args)
        return

    if run and ((not args.distributed or (args.distributed and args.rank == 0))):
        wandb.watch(model, criterion, log="all", log_freq=100)
            



    print("Start training")
    start_time = time.time()
    best_performance = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, run=run)
        lr_scheduler.step()

        if (epoch + 1) % args.lr_drop == 0:
            if args.output_dir != '':
                checkpoint_path = output_dir / f'checkpoint_beforedrop_epoch{epoch:04}.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # Evaluate first and then save the checkpoint to prevent the checkpoint from being lost due to an accident during evaluate
        test_stats = evaluate_hoi(args.dataset_file, model, postprocessors, data_loader_val, args.subject_category_id, device, output_dir, epoch, args)

        if run:
            log_dict = test_stats.copy()
            log_dict['epoch'] = epoch
            run.log(log_dict)
            
        if args.output_dir != '':
            checkpoint_path = output_dir / 'checkpoint_last.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

        if args.dataset_file == 'hico':
            performance = test_stats['mAP_def']
        elif args.dataset_file == 'vcoco':
            performance = test_stats['mAP_all']

        if performance > best_performance and args.output_dir != '':
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'{k}': v for k, v in train_stats.items()},
                     **{f'{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if utils.get_rank() == 0:
        logger.info('\nTraining time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SOV-STG script', parents=[Options.get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

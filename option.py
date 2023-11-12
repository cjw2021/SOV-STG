# ------------------------------------------------------------------------
# SOV-STG
# Copyright (c) 2023 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import argparse
from util.slconfig import DictAction
class Options:
    def get_args_parser():
        parser = argparse.ArgumentParser('Set transformer', add_help=False)
        parser.add_argument('--config_file', '-c', type=str, required=True)
        parser.add_argument('--teacher_config_file', '-t', type=str)
        parser.add_argument('--options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file.')

        # dataset parameters
        parser.add_argument('--dataset_file', default='hico',choices=['hico', 'vcoco'])
        parser.add_argument('--hoi_path', default='./data/hico_det', type=str)
        parser.add_argument('--diff_path', default='./data/hico_diffusion', type=str)
        parser.add_argument('--num_verb_classes', type=int, default=117,
                            help="Number of verb classes")
        parser.add_argument('--num_hoi_classes', type=int, default=600,
                            help="Number of verb classes")
        parser.add_argument('--num_obj_classes', type=int, default=80, help="Number of object classes")
        parser.add_argument('--subject_category_id', default=0, type=int)

        # training parameters
        parser.add_argument('--output_dir', default='',
                            help='path where to save, empty for no saving')
        parser.add_argument('--note', default='',
                            help='add some notes to the experiment')
        parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
        parser.add_argument('--pretrain_model_path', type=str, default='', help='load from other checkpoint')
        parser.add_argument('--swin_pretrained', type=str, default='', help='load swin transformer checkpoint as pretrained backbone weights')
        parser.add_argument('--finetune_ignore', type=str, nargs='+')
        parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                            help='start epoch')
        parser.add_argument('--eval', action='store_true')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=10, type=int)
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--debug', action='store_true')
        parser.add_argument('--find_unused_params', action='store_true')
        parser.add_argument('--use_checkpoint', action='store_true')
        
        # logger
        parser.add_argument('--save_results', action='store_true')
        parser.add_argument('--save_log', action='store_true')
        parser.add_argument('--wo_log_unscaled', action="store_true")
        parser.add_argument('--use_wandb', action="store_true")
        parser.add_argument('--wandb_project', default='HOI_exp', type=str)
        parser.add_argument('--wandb_name', default='train001', type=str)

        # distributed training parameters
        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
        parser.add_argument('--rank', default=0, type=int,
                            help='number of distributed processes')
        parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
        parser.add_argument('--amp', action='store_true',
                            help="Train with mixed precision")
        return parser
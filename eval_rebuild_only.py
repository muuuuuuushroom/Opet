import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
from datasets import build_dataset
import util.misc as utils

from models import build_model
from _backups.engine_ev2 import evaluate

from util.custom_log import *


def get_args_parser():
    parser = argparse.ArgumentParser('Set Point Query Transformer', add_help=False)
    
    parser.add_argument('--cfg', default='outputs/soy_newran/base_pet/config.yaml')
    parser.add_argument('--gt_determined', default='10000', help='test')
    # misc parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='outputs/soy_newran/base_pet/best_checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis_dir', default=None)
    parser.add_argument('--eval_pad', default='padding_center')
    parser.add_argument('--eval_robust', default=[])
    parser.add_argument('--robust_para', default=None)
    parser.add_argument('--prob_map_lc', default=None)

    # 新增：是否启用按 shape 过滤的加载（避免 size mismatch 报错）
    parser.add_argument('--safe_load', action='store_true',
                        help='filter ckpt state_dict by key+shape before loading (avoid size mismatch)')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser


def _filter_state_dict_by_shape(model, ckpt_state):
    """只保留在当前模型中存在且shape一致的权重，避免 size mismatch."""
    model_state = model.state_dict()
    filtered = {}
    skipped = []  # (k, ckpt_shape, model_shape or None, reason)

    for k, v in ckpt_state.items():
        if k not in model_state:
            skipped.append((k, tuple(v.shape), None, "missing_in_model"))
            continue
        if model_state[k].shape != v.shape:
            skipped.append((k, tuple(v.shape), tuple(model_state[k].shape), "shape_mismatch"))
            continue
        filtered[k] = v

    return filtered, skipped


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # build dataset
    val_image_set = 'val'
    dataset_val = build_dataset(image_set=val_image_set, args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
    val_batch_size = 1  # if args.dataset_file == 'RTC' else 4
    data_loader_val = DataLoader(dataset_val, val_batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    # load pretrained model
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        ckpt_state = checkpoint.get('model', checkpoint)

        if getattr(args, "safe_load", False):
            ckpt_state, skipped = _filter_state_dict_by_shape(model_without_ddp, ckpt_state)
            print(f"[safe_load] will load {len(ckpt_state)} tensors, skip {len(skipped)} tensors")
            # 只打印前若干个，避免刷屏
            for k, ck, ms, reason in skipped[:30]:
                print(f"[safe_load][skip] {reason}: {k} ckpt={ck} model={ms}")
            if len(skipped) > 30:
                print(f"[safe_load] ... and {len(skipped)-30} more skipped keys")

        missing, unexpected = model_without_ddp.load_state_dict(ckpt_state, strict=False)
        print(f"load successfully from ckpt: {args.resume}")
        if missing:
            print(f"[load] missing_keys: {len(missing)}")
        if unexpected:
            print(f"[load] unexpected_keys: {len(unexpected)}")

        cur_epoch = checkpoint['epoch'] - 1 if 'epoch' in checkpoint else 0

    # evaluation
    vis_dir = None if args.vis_dir == "" else args.vis_dir
    if vis_dir != None:
        if os.path.exists(vis_dir):
            import shutil
            shutil.rmtree(vis_dir)
        os.makedirs(vis_dir, exist_ok=True)  
    
    import time
    t1 = time.time()
    test_stats = evaluate(model, data_loader_val, device, vis_dir=vis_dir, args=args, criterion=criterion)
    t2 = time.time()
    
    infer_time = t2 - t1
    fps = len(dataset_val) / infer_time 
    print(args, f"\n\ninfer from: {args.output_dir}\ninferring time: {infer_time:.4f}s")
    print(f'FPS: {fps:.2f}')
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('params:', n_parameters/1e6)
    # mae, mse = test_stats['mae'], test_stats['mse']
    # line = f"epoch: {cur_epoch}"  # , mae: {mae}, mse: {mse}, r2: {test_stats['r2']}, rmae: {test_stats['rmae']} "
    print(f'epoch: {cur_epoch}\t\t\tgt > {args.gt_determined} ended with \'ac\' below:')
    
    name = args.resume.split('/')[-2]
    import json
    with open(f"{name}_results.json", "w", encoding="utf-8") as f:
        json.dump(test_stats, f, ensure_ascii=False, indent=4)
    # count = 0
    # for k, v in test_stats.items():
    #     if count % 2 == 0:
    #         print(k,'\t', v, end='\t')
    #     else:
    #         print(k,'\t', v)
    #     count += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser('PET evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    config = load_config(args.cfg)
    args = update_args_with_config(args, config)
    
    args.opt_query_con = False if not hasattr(args, 'opt_query_con') else args.opt_query_con
    if hasattr(args, 'one_key_hfy'):
        if args.one_key_hfy == True:
            args.use_spatial_attention=True
            args.use_arc=True
            args.upsample_strategy='dysample' # dysample, bilinear
            args.fpn_type='panet'  # panet, original
    else:
        args.use_spatial_attention=False
        args.use_arc=False
        args.upsample_strategy='bilinear' # dysample, bilinear
        args.fpn_type='original'  # panet, original
    
    main(args)

"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
from util.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, vis_dir, split_map=None):
    """
    Visualize predictions
    """
    gts = [t['points'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        # draw ground-truth points (red)
        size = 2
        for t in gts[idx]:
            sample_vis = cv2.circle(sample_vis, (int(t[1]), int(t[0])), size, (0, 0, 255), -1)

        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.9 + sample_vis
        
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            name = targets[idx]['image_path'].split('/')[-1].split('.')[0]
            cv2.imwrite(os.path.join(vis_dir, '{}_gt{}_pred{}.jpg'.format(name, len(gts[idx]), len(pred[idx]))), sample_vis)

def _to_device(v, device):
    if torch.is_tensor(v):
        return v.to(device, non_blocking=True)
    # 兼容 list/tuple 里嵌 tensor 的情况（可选）
    if isinstance(v, (list, tuple)):
        return type(v)(_to_device(x, device) for x in v)
    return v

# training
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        targets = [{k: _to_device(v, device) for k, v in t.items()} for t in targets]
        
        # for t in targets:
        #     if "points" in t and torch.is_tensor(t["points"]) and t["points"].shape[0] == 0:
        #         # count 强制为 0（如果存在）
        #         if "count" in t:
        #             if torch.is_tensor(t["count"]):
        #                 t["count"] = torch.zeros_like(t["count"])
        #             else:
        #                 t["count"] = 0

        #         # density 的 999/哨兵值改成安全值（如果存在）
        #         if "density" in t:
        #             if torch.is_tensor(t["density"]):
        #                 t["density"] = torch.zeros_like(t["density"])
        #             else:
        #                 t["density"] = 0.0

        #         # 可选：给个标记，方便后面在 criterion 里分支处理
        #         t["empty_points"] = True
        #     else:
        #         t["empty_points"] = False
        
        gt_points = [target['points'] for target in targets]

        try:
            outputs = model(samples, epoch=epoch, train=True,
                            criterion=criterion, targets=targets)
        except FloatingPointError as e:
            # dump this batch for repro
            os.makedirs("nan_debug", exist_ok=True)
            rank = int(os.environ.get("RANK", -1))

            # collect light meta (avoid huge tensors in print)
            targets_meta = []
            for i, t in enumerate(targets):
                m = {"i": i, "keys": list(t.keys())}
                if "points" in t and torch.is_tensor(t["points"]):
                    m["points_shape"] = tuple(t["points"].shape)
                    m["points_len"] = int(t["points"].shape[0])
                    m["points_nan"] = bool(torch.isnan(t["points"]).any().item())
                    m["points_inf"] = bool(torch.isinf(t["points"]).any().item())
                for k in ("count", "density", "image_id", "file_name"):
                    if k in t:
                        m[k] = t[k]
                targets_meta.append(m)
            print("[NaN-debug] targets_meta (all):", targets_meta)

            # samples stats
            x = samples.tensors.detach().cpu()
            x_clean = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            samp_stats = {
                "shape": tuple(x.shape),
                "min": float(x_clean.min().item()),
                "max": float(x_clean.max().item()),
                "mean": float(x_clean.mean().item()),
                "nan": bool(torch.isnan(x).any().item()),
                "inf": bool(torch.isinf(x).any().item()),
            }

            path = f"nan_debug/train_batch.rank{rank}.epoch{epoch}.pt"
            torch.save(
                {
                    "exception": repr(e),
                    "epoch": epoch,
                    "rank": rank,
                    "samples_tensors": x,          # (bs,3,H,W) CPU
                    "samples_mask": samples.mask.detach().cpu() if hasattr(samples, "mask") and samples.mask is not None else None,
                    "targets": [{k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets],
                    "targets_meta": targets_meta,
                    "samples_stats": samp_stats,
                },
                path,
            )
            raise
        loss_dict, weight_dict, losses = outputs['loss_dict'], outputs['weight_dict'], outputs['losses']

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]

        # inference
        outputs = model(samples, test=True, targets=targets)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]
        
        # process predicted points
        predict_cnt = len(outputs_scores)
        gt_cnt = targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda()
        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        metric_logger.update(mae=results['mae'], mse=results['mse'])

        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'])

        # visualize predictions
        if vis_dir: 
            points = [[point[0]*img_h, point[1]*img_w] for point in outputs_points]     # recover to actual points
            split_map = (outputs['split_map_raw'][0].detach().cpu().squeeze(0) > 0.5).float().numpy()
            visualization(samples, targets, [points], vis_dir, split_map=split_map)
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results['mse'])
    return results

            


@torch.no_grad()
def evaluate_tile(model, data_loader, device, epoch=0, vis_dir=None, tile_size=1024, overlap=128):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    print_freq = 10

    def gen_tiles_coords(H, W, tile_size, overlap):
        step = tile_size - overlap
        ys = list(range(0, max(1, H - tile_size + 1), step))
        xs = list(range(0, max(1, W - tile_size + 1), step))
        if ys and ys[-1] + tile_size < H:
            ys.append(H - tile_size)
        if xs and xs[-1] + tile_size < W:
            xs.append(W - tile_size)
        if not ys:
            ys = [0]
        if not xs:
            xs = [0]
        coords = []
        for y in ys:
            for x in xs:
                y1 = min(y + tile_size, H)
                x1 = min(x + tile_size, W)
                coords.append((y, x, y1, x1))
        return coords

    def run_on_tile(tile_tensor, model, device):
        # tile_tensor: C,H,W (cpu tensor, already normalized)
        # return: list of predicted points in normalized coords relative to tile (N,2) and scores
        # build NestedTensor
        mask = torch.zeros((1, tile_tensor.shape[1], tile_tensor.shape[2]), dtype=torch.bool)
        nested = NestedTensor(tile_tensor.unsqueeze(0), mask.unsqueeze(0))  # 1,C,H,W
       # mask must be (H, W). NestedTensor expects mask shape (N, H, W),
        # so expand to (1, H, W) when creating NestedTensor.
        mask = torch.zeros((tile_tensor.shape[1], tile_tensor.shape[2]), dtype=torch.bool)  # (H, W)
        nested = NestedTensor(tile_tensor.unsqueeze(0), mask.unsqueeze(0))  # mask -> (1, H, W)
        nested = nested.to(device)
        with torch.no_grad():
            try:
                autocast = torch.cuda.amp.autocast
            except AttributeError:
                autocast = None
            if autocast is not None and device.type == 'cuda':
                with autocast():
                    outputs = model(nested, test=True, targets=None)
            else:
                outputs = model(nested, test=True, targets=None)
         # outputs_points likely shape (1, N, 2) normalized; outputs_scores (1,N)
        out_pts = outputs['pred_points'][0].detach().cpu()  # N,2
        out_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0].detach().cpu()
        return out_pts, out_scores, outputs

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # samples: NestedTensor with batch size=1 in val
        samples = samples.to(device)
        img_h, img_w = samples.tensors.shape[-2:]
        # If image fits in one tile -> use original flow
        if img_h <= tile_size and img_w <= tile_size:
            # original single-tile inference
            outputs = model(samples, test=True, targets=targets)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            # convert to absolute coords
            points_abs = [[float(p[0] * img_h), float(p[1] * img_w)] for p in outputs_points]
        else:
            # tiled inference
            # get the raw image tensor from samples (normalized C,H,W)
            img_tensor = samples.tensors[0].cpu()  # C,H,W
            tiles = gen_tiles_coords(img_h, img_w, tile_size, overlap)
            all_preds = []
            all_scores = []
            for (y0, x0, y1, x1) in tiles:
                tile = img_tensor[:, y0:y1, x0:x1].contiguous()
                th, tw = tile.shape[1], tile.shape[2]
                # if tile smaller than tile_size, it is ok (no padding)
                out_pts, out_scores, _ = run_on_tile(tile, model, device)
                # out_pts normalized relative to tile -> convert to absolute coords
                for i in range(out_pts.shape[0]):
                    py_norm, px_norm = float(out_pts[i][0]), float(out_pts[i][1])
                    py = py_norm * th + y0
                    px = px_norm * tw + x0
                    all_preds.append((py, px))
                    all_scores.append(float(out_scores[i]))
            # simple deduplication: round to int and unique
            rounded = {}
            for (py, px), sc in zip(all_preds, all_scores):
                key = (int(round(py)), int(round(px)))
                if key in rounded:
                    # keep higher score
                    if sc > rounded[key]:
                        rounded[key] = sc
                else:
                    rounded[key] = sc
            points_abs = [[float(k[0]), float(k[1])] for k in rounded.keys()]

        # process predicted points
        predict_cnt = len(points_abs)
        gt_cnt = targets[0]['points'].shape[0]

        # compute error
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)

        # record results
        results = {}
        toTensor = lambda x: torch.tensor(x).float().cuda() if device.type == 'cuda' else torch.tensor(x).float()
        results['mae'], results['mse'] = toTensor(mae), toTensor(mse)
        metric_logger.update(mae=results['mae'], mse=results['mse'])

        results_reduced = utils.reduce_dict(results)
        metric_logger.update(mae=results_reduced['mae'], mse=results_reduced['mse'])

        # visualize predictions (use stitched points)
        if vis_dir:
            # create fake outputs dict similar to original so visualization can use it
            # build outputs_points normalized to full image for visualization
            out_pts_full = []
            for p in points_abs:
                out_pts_full.append([p[0] / img_h, p[1] / img_w])
            split_map = None
            # reuse visualization with constructing minimal structures
            class SimpleOutputs:
                pass
            simple_out = {'pred_points': [torch.tensor(out_pts_full)], 'split_map_raw': [torch.zeros(1,1,1)], 'pred_logits': torch.zeros(1,1,2)}
            # visualization expects samples (NestedTensor) and targets etc.
            # convert list-of-lists to expected format used in visualization
            visualization(samples, targets, [points_abs], vis_dir, split_map=split_map)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['mse'] = np.sqrt(results['mse'])
    return results
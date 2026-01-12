"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
from sklearn.metrics import r2_score
import numpy as np
import cv2

import torch
import torchvision.transforms as standard_transforms
import torch.nn.functional as F

import util.misc as utils
import util.metrics as metrics
from util.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def visualization(samples, targets, pred, queries, vis_dir, gt_cnt, split_map=None, args=None, outputs=None):
    """                         # pred point & query
    Visualize predictions
    """
    box_flag = True if args.opt_query_decoder else False
    
    if args.dataset_file != 'WuhanMetro':
        vis_dir_ac = os.path.join(vis_dir, 'gt_ac')
        vis_dir_nac = os.path.join(vis_dir, 'gt_nac')
        os.makedirs(vis_dir_ac, exist_ok=True)
        os.makedirs(vis_dir_nac, exist_ok=True)
    else:
        vis_dir_ac = vis_dir
        vis_dir_nac = vis_dir
        os.makedirs(vis_dir + '/3-1', exist_ok=True)
        os.makedirs(vis_dir + '/3-2', exist_ok=True)
    
    if box_flag:
        vis_dir_box = os.path.join(vis_dir, 'box')
        os.makedirs(vis_dir_box, exist_ok=True)
        # vis_dir_box_nosame = os.path.join(vis_dir, 'box_nosame')
        # os.makedirs(vis_dir_box_nosame, exist_ok=True)
    
    gts = [t["points"].tolist() for t in targets]                   # gt point         
    
    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose(
        [
            DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            standard_transforms.ToPILImage(),
        ]
    )

    images = samples.tensors
    masks = samples.mask
    for idx in range(images.shape[0]):
        sample = restore_transform(images[idx])
        sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
        sample_vis = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        or_sample = sample_vis
        h, w = sample_vis.shape[:2]
        # sample_vis = cv2.resize(sample_vis, (512, 512))

        if box_flag:
            pq_ex = outputs['pq_ex']
            sample_vis_box = or_sample
            batch_size, num_points, _ = pq_ex.shape
            
            fixed_colors = [
                # (0, 0, 255),      # 红
                # (0, 128, 255),    # 橙
                # (0, 255, 255),    # 黄
                (0, 255, 0),      # 绿
                # (255, 255, 0),    # 青
                (255, 0, 0),      # 蓝
                (255, 0, 255),    # 紫
            ]
            
            # for i, p in enumerate(pred[idx]):
            #     sample_vis_box = cv2.circle(
            #         sample_vis_box, (int(p[1]), int(p[0])), 4, (255, 0, 255), -1
            #         )
            for i, t in enumerate(gts[idx]):
                sample_vis_box = cv2.circle(
                    sample_vis_box, (int(t[1]), int(t[0])), 3, (0, 0, 255), -1
                )
            
            for i in range(num_points):  # 遍历每个点序号
                for b in range(batch_size):  # 每个 batch 的该点
                    q = pq_ex[b][i].clone()
                    q[1] *= w
                    q[0] *= h
                    q_x, q_y = int(q[1]), int(q[0])
                    color = fixed_colors[b % len(fixed_colors)]
                    sample_vis_box = cv2.circle(sample_vis_box, (q_x, q_y), 2, color, -1)
                    
            for i, p in enumerate(pred[idx]):
                sample_vis_box = cv2.circle(
                    sample_vis_box, (int(p[1]), int(p[0])), 2, (255, 0, 255), -1
                    )

        
            name = targets[idx]["image_path"].split("/")[-1].split(".")[0]
            # start_points = pq_ex[0]      # shape: [num_points, 2]
            # end_points   = pq_ex[-1]
            # offsets = end_points - start_points
            # if offsets.shape[0] != 0:
            # ref_offset = offsets[0]
            # is_equal = torch.all(offsets == ref_offset, dim=1) 
            # same_ratio = is_equal.float().mean().item()
            save_path = os.path.join(
                    vis_dir_box,
                    "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                ) 
            cv2.imwrite(
                save_path,
                sample_vis_box,
            )
            
        # if not box_flag:
        # draw ground-truth points (red)
        size = 2
        for i, t in enumerate(gts[idx]):
            sample_vis = cv2.circle(
                sample_vis, (int(t[1]), int(t[0])), size+2, (0, 0, 255), -1
            )
                    # cv2.putText(sample_vis, str(i+1), (int(t[1]+3), int(t[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw predictions (green)
        for i, p in enumerate(pred[idx]):
            sample_vis = cv2.circle(
                sample_vis, (int(p[1]), int(p[0])), size+1, (0, 255, 0), -1
                )
                    # cv2.putText(sample_vis, str(i+1), (int(p[1]+3), int(p[0])+3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # draw point-query
        for i, q in enumerate(queries[idx]):
            
            if args.predict == 'origin':
                q[1] *= w
                q[0] *= h
            sample_vis = cv2.circle(
                sample_vis, (int(q[1]), int(q[0])), size, (0, 255, 255), -1
                )
            # draw line between query and pred
            q_x, q_y = int(q[1]), int(q[0])
            p_x, p_y = int(pred[idx][i][1]), int(pred[idx][i][0])
            overlay = sample_vis.copy()
            cv2.line(overlay, (p_x, p_y), (q_x, q_y), (0, 255, 0), 2) 
            alpha = 0.5 
            sample_vis = cv2.addWeighted(overlay, alpha, sample_vis, 1 - alpha, 0)
            # cv2.putText(sample_vis, str(i+1), (int(q[1]*w-3), int(q[0]*h)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # draw split map
        if split_map is not None:
            imgH, imgW = sample_vis.shape[:2]
            split_map = (split_map * 255).astype(np.uint8)
            split_map = cv2.applyColorMap(split_map, cv2.COLORMAP_JET)
            split_map = cv2.resize(split_map, (imgW, imgH), interpolation=cv2.INTER_NEAREST)
            sample_vis = split_map * 0.7 + sample_vis

        # save image
        if vis_dir is not None:
            # eliminate invalid area
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[: valid_h + 1, : valid_w + 1]

            if args.dataset_file == 'WuhanMetro':
                pre_name_folder = targets[idx]["image_path"].split("/")[-2]
            name = targets[idx]["image_path"].split("/")[-1].split(".")[0]
            
            if gt_cnt > 100:
                cv2.imwrite(
                    os.path.join(
                        vis_dir_ac,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )
            else:
                if args.dataset_file == 'WuhanMetro':
                    vis_dir_nac = os.path.join(vis_dir, pre_name_folder)
                cv2.imwrite(
                    os.path.join(
                        vis_dir_nac,
                        "{}_gt{}_pred{}.jpg".format(name, len(gts[idx]), len(pred[idx])),
                    ),
                    sample_vis,
                )

# evaluation
@torch.no_grad()
def evaluate(model, data_loader, device, epoch=0, vis_dir=None, distributed=False, args=None, criterion=None): 
    model.eval()

    gt_determined = 1 if args.dataset_file == 'WuhanMetro' else 100
    # gt_determined = 100
    metric_logger = utils.MetricLogger(delimiter="  ", win_size=len(data_loader))
    header = 'Test:'

    # if vis_dir is not None:
        # os.makedirs(vis_dir, exist_ok=True)
        # vis_dir_ac = os.path.join(vis_dir, 'gt_ac')
        # os.makedirs(vis_dir_ac, exist_ok=True)
        
    gt_cnt_all, pd_cnt_all = [], []
    gt_cnt_all_ac, pd_cnt_all_ac = [], []
    y_pred_all = []
    results = {}
    print_freq = 10; count = 0
    for samples, img_path, _ in metric_logger.log_every(data_loader, print_freq, header):
        
        samples = samples.to(device)    # tensors [torch.Size([1, 3, 512, 1024])]
        img_h, img_w = samples.tensors.shape[-2:]     
        outputs = model(samples, test=True, targets=None, criterion=criterion, eval_s=True)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]
        outputs_offsets = outputs['pred_offsets'][0]
        outputs_queries = outputs['points_queries']

        # process predicted points
        predict_cnt = len(outputs_scores)

        # compute error
        # record results
        predict_cnt = len(outputs_scores)
        y_pred_all.append(predict_cnt)
        
        key_name = img_path[0].split('/')[-1].replace('.jpg', '')
        results[key_name] = predict_cnt

 
    metric_logger.synchronize_between_processes()

    # results["rmae"], results["rmse"] = metrics.compute_relerr(
    #     gt_cnt_array, pd_cnt_array
        

    return results

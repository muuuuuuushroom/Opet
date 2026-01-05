import gradio as gr
import torch
import argparse
from PIL import Image
from models import build_model
from datasets.SOY import build_soy
from engine_evon import DeNormalize

from util.custom_log import *
import os
import torchvision.transforms as standard_transforms
import cv2
import numpy as np
from util.misc import nested_tensor_from_tensor_list  # add

global_model = None
global_args = None
global_transform = None  

def load_model(cfg_path, device):
    config = load_config(cfg_path)
    config['resume'] = '/data/zlt/Projects/PET_find/Opet/outputs/soy_newran/base_pet/best_checkpoint.pth'
    args = argparse.Namespace(**config)
    args.resume = args.resume or None
    model, _ = build_model(args)
    
    # 检查 checkpoint 文件路径是否有效
    if args.resume is None or not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
    
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, args

def initialize_model(cfg_path, checkpoint_path):
    """初始化模型并存储到全局变量中"""
    global global_model, global_args, global_transform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config(cfg_path)
    config['resume'] = checkpoint_path  # 确保将 checkpoint_path 设置到配置中
    global_model, global_args = load_model(cfg_path, device)
    
    dataset = build_soy(image_set='val', args=global_args)
    global_transform = dataset.transform
    print('Model and dataset loaded successfully')
    
def visualization(samples, pred, vis_dir):
    """
    Visualize predictions
    """
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
        h, w = sample_vis.shape[:2]
        # draw ground-truth points (red)
        size = 3
        # draw predictions (green)
        for p in pred[idx]:
            sample_vis = cv2.circle(sample_vis, (int(p[1]), int(p[0])), size, (0, 255, 0), -1)
            
        # save image
        if vis_dir is not None:
            # eliminate invalid area
            imgH, imgW = masks.shape[-2:]
            valid_area = torch.where(~masks[idx])
            valid_h, valid_w = valid_area[0][-1], valid_area[1][-1]
            sample_vis = sample_vis[:valid_h+1, :valid_w+1]

            cv2.imwrite(os.path.join(vis_dir, 'example.jpg'), sample_vis)

def predict(image):
    """使用全局加载的模型进行预测
    Gradio inputs: gr.Image(type="filepath") -> `image` is a filepath (str)
    """
    global global_model, global_args, global_transform
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) read image from filepath
    pil_img = Image.open(image).convert("RGB")

    # 2) pad/resize to be divisible by base_size
    w, h = pil_img.size
    base_size = 256
    new_w = (w + base_size - 1) // base_size * base_size
    new_h = (h + base_size - 1) // base_size * base_size
    if (new_w, new_h) != (w, h):
        pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    print(f"Adjusted image size: {pil_img.size}")

    # 3) PIL -> tensor
    pil_to_tensor = standard_transforms.ToTensor()
    tensor_image = pil_to_tensor(pil_img).to(device)  # [C,H,W], float

    # 3.5) tensor -> NestedTensor (model input)
    samples = nested_tensor_from_tensor_list([tensor_image]).to(device)

    with torch.no_grad():
        outputs = global_model(samples)  # was: global_model(tensor_image.unsqueeze(0))

    # 4) simple visualization on the resized image (avoid NestedTensor dependency)
    vis_dir = "./visualizations_cache"
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, "example.jpg")

    vis_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    pred_points = outputs.get("pred_points", None)
    if pred_points is not None:
        pts = pred_points[0]
        if isinstance(pts, torch.Tensor):
            pts = pts.detach().cpu().numpy()
        for p in pts:
            vis_bgr = cv2.circle(vis_bgr, (int(p[1]), int(p[0])), 3, (0, 255, 0), -1)

    cv2.imwrite(vis_path, vis_bgr)
    return vis_path

# Gradio 界面
def gradio_demo():
    demo = gr.Interface(
        fn=predict,
        inputs=[
            gr.Image(type="filepath", label="上传图片"),
        ],
        outputs=gr.Image(type="filepath", label="预测结果"),  # 修改为图像输出
        title="模型预测 Demo",
        description="上传图片，实时获取模型预测结果"
    )
    demo.launch()

if __name__ == "__main__":
    # 在程序启动时加载模型
    initialize_model(
        cfg_path='/data/zlt/Projects/PET_find/Opet/outputs/soy_newran/base_pet/config.yaml', 
        checkpoint_path='/data/zlt/Projects/PET_find/Opet/outputs/soy_newran/base_pet/best_checkpoint.pth'
    )
    gradio_demo()

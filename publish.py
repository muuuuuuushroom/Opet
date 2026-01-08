import gradio as gr
import torch
import argparse
from PIL import Image
from models import build_model
from engine_evon import DeNormalize

from util.custom_log import *
import os
import torchvision.transforms as standard_transforms
import cv2
import numpy as np
from util.misc import nested_tensor_from_tensor_list 
import zipfile
import tempfile
from pathlib import Path
import shutil
import pandas as pd

global_model = None
global_args = None
global_transform = None  
global_criterion = None

def load_model(cfg_path, device, ckpt_path):
    config = load_config(cfg_path)
    args = argparse.Namespace(**config)
    args.resume = ckpt_path
    model, criterion = build_model(args)
    
    if args.resume is None or not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
    
    checkpoint = torch.load(args.resume, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model, args, criterion

def initialize_model(cfg_path, checkpoint_path):
    global global_model, global_args, global_transform, global_criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model, global_args, global_criterion = load_model(cfg_path, device, checkpoint_path)
    # dataset = build_soy(image_set='val', args=global_args)
    # global_transform = None  # dataset.transform
    print('Model and dataset loaded successfully')
    
def _maybe_free_infer_cuda_memory():
    """释放推理过程中产生的临时显存缓存；不影响已加载的model常驻显存。"""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # 可选：更激进的回收（通常不必，但你已有在OOM里用它）
            torch.cuda.ipc_collect()
        except Exception:
            pass

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

            cv2.imwrite(os.path.join(vis_dir, 'single/example.jpg'), sample_vis)

def _handle_oom(e: Exception, context: str):
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    raise gr.Error(f"{context}：CUDA OOM（显存不足）。请尝试换小图/减少并发/改用CPU。原始信息：{e}")

def predict(image):
    """
    Gradio inputs: gr.Image(type="filepath") -> `image` is a filepath (str)
    """
    global global_model, global_args, global_transform, global_criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # read image from filepath
    pil_img = Image.open(image).convert("RGB")

    pil_to_tensor = standard_transforms.ToTensor()
    tensor_image = pil_to_tensor(pil_img)

    # torchvision Normalize ImageNet
    normalize = standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    tensor_image = normalize(tensor_image).to(device) # [C,H,W], float

    samples = nested_tensor_from_tensor_list([tensor_image]).to(device)

    try:
        with torch.no_grad():
            outputs = global_model(samples, test=True, targets=None)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            outputs_offsets = outputs['pred_offsets'][0]
            outputs_queries = outputs['points_queries']
        predict_cnt = len(outputs_scores)
        print('Total Counts:', predict_cnt)
        counts_text = f"计数值： {predict_cnt}"
        
        # visualization
        vis_dir = "./visualizations_cache"
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, "example.jpg")

        vis_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pred_points = outputs.get("pred_points", None)
        
        img_h, img_w = samples.tensors.shape[-2:]
        if pred_points is not None:
            pts = pred_points[0]
            pts = [[pt[0]*img_h, pt[1]*img_w] for pt in pts] 
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
                
            for p in pts:
                vis_bgr = cv2.circle(vis_bgr, (int(p[1]), int(p[0])), 3, (0, 0, 255), -1)

        cv2.imwrite(vis_path, vis_bgr)
        return vis_path, counts_text
    except torch.cuda.OutOfMemoryError as e:
        _handle_oom(e, "单图推理失败")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _handle_oom(e, "单图推理失败")
        raise
    finally:
        # 释放本次推理产生的临时显存（不动global_model）
        try:
            del samples, tensor_image
        except Exception:
            pass
        try:
            del outputs
        except Exception:
            pass
        try:
            del outputs_scores
        except Exception:
            pass
        _maybe_free_infer_cuda_memory()

def _count_from_pil(pil_img: Image.Image) -> int:
    global global_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pil_to_tensor = standard_transforms.ToTensor()
    tensor_image = pil_to_tensor(pil_img.convert("RGB"))

    normalize = standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    tensor_image = normalize(tensor_image).to(device)
    samples = nested_tensor_from_tensor_list([tensor_image]).to(device)

    outputs = None
    pts = None
    try:
        with torch.no_grad():
            outputs = global_model(samples, test=True, targets=None)
        img_h, img_w = samples.tensors.shape[-2:]
        pred_points = outputs.get("pred_points", None)
        if pred_points is not None:
            _pts = pred_points[0]
            pts = [[pt[0]*img_h, pt[1]*img_w] for pt in _pts]
        return outputs, pts
    finally:
        # 只清理输入/中间变量；outputs要返回给上层就不del它
        try:
            del samples, tensor_image
        except Exception:
            pass
        _maybe_free_infer_cuda_memory()

def predict_zip(zip_path):
    """
    Gradio inputs: gr.File -> `zip_path` typically is a tempfile.NamedTemporaryFile-like.
    Returns:
      - excel filepath (.xlsx)
      - visualizations zip filepath (.zip)
      - dataframe rows: [[filename, count], ...]
    """
    if hasattr(zip_path, "name"):
        zip_path = zip_path.name

    if not isinstance(zip_path, str) or not os.path.isfile(zip_path):
        return None, None, []

    if not zip_path.lower().endswith(".zip"):
        return None, None, []

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    base_vis_dir = Path("./visualizations_cache/from_zips")
    if base_vis_dir.exists():
        for p in base_vis_dir.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
    base_vis_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total = 0
    oom_count = 0
    with tempfile.TemporaryDirectory(prefix="gr_zip_") as tmpdir:
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmpdir)
        except Exception as e:
            return None, None, []

        files = [p for p in Path(tmpdir).rglob("*") if p.is_file() and p.suffix.lower() in exts]
        files.sort(key=lambda p: str(p).lower())

        if not files:
            return None, None, []

        for p in files:
            outputs = None
            outputs_scores = None
            pil_img = None
            pts = None
            try:
                pil_img = Image.open(str(p)).convert("RGB")
                outputs, pts = _count_from_pil(pil_img)
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

                cnt = int(len(outputs_scores))
                
                stem = Path(p.name).stem
                out_name = f"{stem}_pred{cnt}.jpg"
                out_path = base_vis_dir / out_name
                k = 1
                while out_path.exists():
                    out_name = f"{stem}_pred{cnt}_{k}.jpg"
                    out_path = base_vis_dir / out_name
                    k += 1
                
                # visualization
                vis_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                if pts is not None:
                    if isinstance(pts, torch.Tensor):
                        pts = pts.detach().cpu().numpy()
                    for pt in pts:
                        vis_bgr = cv2.circle(vis_bgr, (int(pt[1]), int(pt[0])), 3, (0, 0, 255), -1)
                cv2.imwrite(str(out_path), vis_bgr)

                rows.append([p.name, cnt])
                total += cnt
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
                rows.append([p.name, "OOM: 显存不足"])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                    rows.append([p.name, "OOM: 显存不足"])
                else:
                    rows.append([p.name, f"失败: {e}"])
            except Exception as e:
                rows.append([p.name, f"失败: {e}"])
            finally:
                # 每张图推理结束就清理一次，避免批量累积显存碎片/缓存
                try:
                    del outputs_scores
                except Exception:
                    pass
                try:
                    del outputs
                except Exception:
                    pass
                _maybe_free_infer_cuda_memory()

    # Export Excel
    excel_path = str(base_vis_dir / "counts.xlsx")
    try:
        df = pd.DataFrame(rows, columns=["文件名", "计数/状态"])
        df.to_excel(excel_path, index=False)
    except Exception as e:
        excel_path = None
        rows.append(["__export__", f"Excel导出失败: {e}"])

    # Export visualization zip
    vis_zip_path = str(base_vis_dir / "visualizations.zip")
    try:
        with zipfile.ZipFile(vis_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for imgp in sorted(base_vis_dir.glob("*.jpg")):
                zf.write(str(imgp), arcname=imgp.name)
    except Exception as e:
        vis_zip_path = None
        rows.append(["__export__", f"可视化打包失败: {e}"])

    return  excel_path, vis_zip_path, rows


USER_CREDENTIALS = {"admin": "654321"}

def check_login(username, password):
    """验证登录信息"""
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True, "登录成功！"
    else:
        return False, "用户名或密码错误！"

def gradio_demo():
    with gr.Blocks(title="大豆胞囊虫计数系统 - 请先登录") as demo:
        # 登录界面（初始显示）
        with gr.Column(visible=True, elem_id="login_section") as login_section:
            gr.Markdown("# 🔐 大豆胞囊虫计数系统")
            gr.Markdown("### 请先登录以使用系统")
            
            with gr.Row():
                with gr.Column(scale=1):
                    username = gr.Textbox(
                        label="用户名", 
                        value="admin",  # 默认用户名
                        placeholder="输入用户名",
                        scale=2
                    )
                with gr.Column(scale=1):
                    password = gr.Textbox(
                        label="密码", 
                        type="password",
                        value="654321",  # 默认密码
                        placeholder="输入密码",
                        scale=2
                    )
            
            with gr.Row():
                login_btn = gr.Button("登录", variant="primary", size="lg")
                clear_btn = gr.Button("清除", size="lg")
            
            login_status = gr.Textbox(label="登录状态", visible=False)
        
        # 主应用界面（初始隐藏）
        with gr.Column(visible=False, elem_id="main_section") as main_section:
            gr.Markdown("# 🌱 大豆胞囊虫计数 Demo")
            
            with gr.Tab("单图精细化点回归计数"):
                with gr.Row():
                    in_img = gr.Image(type="filepath", label="上传图片")
                with gr.Row():
                    clear_btn_main = gr.Button("清除")
                    submit_btn = gr.Button("提交", variant="primary")
                with gr.Row():
                    out_img = gr.Image(type="filepath", label="预测结果")
                with gr.Row():
                    out_txt = gr.Textbox(label="统计信息", lines=1)
                
                submit_btn.click(fn=predict, inputs=[in_img], outputs=[out_img, out_txt])
                clear_btn_main.click(
                    fn=lambda: [None, None, None], 
                    inputs=None, 
                    outputs=[in_img, out_img, out_txt]
                )
            
            with gr.Tab("高通量批量图像分析"):
                zip_in = gr.File(label="上传 .zip 压缩包文件", file_types=[".zip"])
                batch_btn = gr.Button("开始批量计数", variant="primary")
                
                with gr.Row():
                    out_excel = gr.File(label="导出计数报表")
                    out_viszip = gr.File(label="下载所有可视化")
                
                out_table = gr.Dataframe(headers=["文件名", "计数/状态"], label="结果", wrap=True)
                batch_btn.click(
                    fn=predict_zip,
                    inputs=[zip_in],
                    outputs=[out_excel, out_viszip, out_table]
                )
            
            # 添加退出登录按钮
            with gr.Row():
                logout_btn = gr.Button("退出登录", variant="secondary")
        
        # 登录按钮事件
        def login_action(username, password):
            success, message = check_login(username, password)
            if success:
                return [
                    gr.update(visible=False),  # 隐藏登录界面
                    gr.update(visible=True),   # 显示主界面
                    gr.update(value=message, visible=True)
                ]
            else:
                try:
                    gr.Warning(message)
                except Exception:
                    pass

                return [
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value=message, visible=True)
                ]
        
        # 清除按钮事件
        def clear_login():
            return [
                gr.update(value="admin"),
                gr.update(value="654321"),
                gr.update(visible=False)
            ]
        
        # 退出登录事件
        def logout_action():
            return [
                gr.update(visible=True),   # 显示登录界面
                gr.update(visible=False),  # 隐藏主界面
                gr.update(value="", visible=False)
            ]
        
        # 绑定事件
        login_btn.click(
            fn=login_action,
            inputs=[username, password],
            outputs=[login_section, main_section, login_status]
        )
        
        clear_btn.click(
            fn=clear_login,
            inputs=None,
            outputs=[username, password, login_status]
        )
        
        logout_btn.click(
            fn=logout_action,
            inputs=None,
            outputs=[login_section, main_section, login_status]
        )
        
        # 回车键也可以触发登录
        password.submit(
            fn=login_action,
            inputs=[username, password],
            outputs=[login_section, main_section, login_status]
        )
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860, 
        share=False,
        show_error=True,
        debug=True
    )

if __name__ == "__main__":
    
    initialize_model(
        cfg_path='outputs/soy_newran/base_pet333/config.yaml', 
        checkpoint_path='outputs/soy_newran/base_pet333/best_checkpoint.pth'
    )
    # predict('data/soybean/images/24S2983-4-2.jpg')
    gradio_demo()
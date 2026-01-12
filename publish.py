import gradio as gr
import torch
import argparse
from PIL import Image
from models import build_model

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

import time
import uuid

# CFG='pretrained/config.yaml'
# CKPT='pretrained/best_checkpoint.pth'
CFG='outputs/WuhanMetro/base_pet/config.yaml'
CKPT='outputs/WuhanMetro/base_pet/best_checkpoint.pth'

global_model = None
global_args = None
global_transform = None  
global_criterion = None

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

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
        # draw predictions
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
    raise gr.Error(f"{context}：CUDA OOM（显存不足）。请尝试换小图。原始信息：{e}")

def predict(image, session_dir: str, history: list):
    """
    Gradio inputs:
      - image: gr.Image(type="filepath")
      - session_dir: gr.State(str)
      - history: gr.State(list)
    Returns:
      - out_img_path
      - out_txt
      - updated history
      - updated dataframe rows
    """
    global global_model, global_args, global_transform, global_criterion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not image or not isinstance(image, str) or not os.path.isfile(image):
        raise gr.Error("请先上传图片。")

    if not session_dir:
        # 兜底：没有初始化 session 时，临时创建一个
        _, session_dir = _new_session()

    # read image
    pil_img = Image.open(image).convert("RGB")
    w, h = pil_img.size
    if max(w, h) > 3200:
        scale = 1600.0 / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)

    pil_to_tensor = standard_transforms.ToTensor()
    tensor_image = pil_to_tensor(pil_img)

    normalize = standard_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    tensor_image = normalize(tensor_image).to(device)
    samples = nested_tensor_from_tensor_list([tensor_image]).to(device)

    outputs = None
    outputs_scores = None

    try:
        with torch.no_grad():
            outputs = global_model(samples, test=True, targets=None)
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        predict_cnt = int(len(outputs_scores))
        out_txt = f"计数值： {predict_cnt}"

        # 可视化（先生成本次输出图）
        vis_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        pred_points = outputs.get("pred_points", None)

        img_h, img_w = samples.tensors.shape[-2:]
        if pred_points is not None:
            pts = pred_points[0]
            pts = [[pt[0] * img_h, pt[1] * img_w] for pt in pts]
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
            for p in pts:
                vis_bgr = cv2.circle(vis_bgr, (int(p[1]), int(p[0])), 3, (0, 0, 255), -1)

        # === session 缓存：按编号保存输入/输出 ===
        idx = len(history or [])
        sess_dir = Path(session_dir)
        _ensure_dir(sess_dir)

        src_path = Path(image)
        orig_stem = src_path.stem
        orig_suffix = src_path.suffix.lower() or ".jpg"
        
        in_dst = sess_dir / f"{orig_stem}{orig_suffix}"
        out_dst = sess_dir / f"{orig_stem}_pred{predict_cnt}.jpg"
        
        k = 1
        while in_dst.exists() or out_dst.exists():
            in_dst = sess_dir / f"{orig_stem}_{k}{orig_suffix}"
            out_dst = sess_dir / f"{orig_stem}_{k}_pred{predict_cnt}.jpg"
            k += 1

        in_path_saved = _safe_copy(image, in_dst)
        cv2.imwrite(str(out_dst), vis_bgr)
        out_path_saved = str(out_dst)

        item = {
            "idx": idx,
            "ts": _now_tag(),
            "in_img": in_path_saved,
            "out_img": out_path_saved,
            "out_text": out_txt,
            # "in_name": src_path.name, 
        }
        history = _append_history(history, item, limit=50)
        history_df = _history_rows(history)
        gallery_items = [it.get("out_img") for it in (history or []) if it.get("out_img")]

        return out_path_saved, out_txt, history, history_df, gallery_items

    except torch.cuda.OutOfMemoryError as e:
        _handle_oom(e, "单图推理失败")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            _handle_oom(e, "单图推理失败")
        raise
    finally:
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
    # If resolution too large (e.g., >3000x3000), downsample longest side to 1600 while keeping aspect ratio
    w, h = pil_img.size
    if max(w, h) > 3200:
        scale = 1600.0 / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil_img = pil_img.resize((new_w, new_h), resample=Image.BILINEAR)

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

def _now_tag() -> str:
    return time.strftime("%Y.%m.%d-%H.%M.%S")

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _safe_copy(src: str, dst: Path) -> str:
    """复制文件到指定位置并返回新路径；尽量保留元信息。"""
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)

def _session_root() -> Path:
    # 每次打开页面生成一个 session_id；历史仅对本次访问有效
    return Path("visualizations_cache") / "sessions"

def _new_session() -> tuple[str, str]:
    session_id = uuid.uuid4().hex
    session_dir = _session_root() / session_id
    _ensure_dir(session_dir)
    return session_id, str(session_dir)

def _clear_session_dir(session_dir: str):
    try:
        shutil.rmtree(session_dir, ignore_errors=True)
    except Exception:
        pass

def _append_history(history: list, item: dict, limit: int = 50) -> list:
    history = (history or []) + [item]
    if len(history) > limit:
        history = history[-limit:]
    return history

def _history_rows(history: list) -> list[list[str]]:
    # Dataframe 展示：编号、时间、文件名、计数摘要
    rows = []
    for i, it in enumerate(history or []):
        in_name = os.path.basename(it.get("in_img", "") or "")
        rows.append([str(i), it.get("ts", ""), in_name, it.get("out_text", "")])
    return rows

def _on_history_select(evt: gr.SelectData, history: list):
    """
    点击历史行：回填 in_img/out_img/out_txt
    Dataframe 的 evt.index 通常是 (row, col)
    """
    if not history:
        return gr.update(), gr.update(), gr.update()

    idx = evt.index
    if isinstance(idx, (tuple, list)):
        idx = idx[0]
    try:
        idx = int(idx)
    except Exception:
        return gr.update(), gr.update(), gr.update()

    if idx < 0 or idx >= len(history):
        return gr.update(), gr.update(), gr.update()

    it = history[idx]
    return it.get("in_img"), it.get("out_img"), it.get("out_text")

def _on_history_gallery_select(evt: gr.SelectData, history: list):
    """
    点击 Gallery：evt.index 通常是 int
    """
    if not history:
        return gr.update(), gr.update(), gr.update()

    idx = evt.index
    try:
        idx = int(idx)
    except Exception:
        return gr.update(), gr.update(), gr.update()

    if idx < 0 or idx >= len(history):
        return gr.update(), gr.update(), gr.update()

    it = history[idx]
    return it.get("in_img"), it.get("out_img"), it.get("out_text")


def export_single_history(session_dir: str, history: list):
    """
    导出本次会话单图历史：
      - counts.xlsx: 输入文件名、计数/状态（从 out_text 解析）
      - visualizations.zip: 历史预测输出图（out_img）
    Returns: (excel_path, zip_path)
    """
    if not session_dir:
        return None, None
    if not history:
        raise gr.Error("本次会话还没有历史记录，无法导出。")

    sess_dir = Path(session_dir)
    _ensure_dir(sess_dir)

    # 1) Excel
    rows = []
    for it in (history or []):
        in_name = os.path.basename(it.get("in_img", "") or "")
        out_text = (it.get("out_text", "") or "").strip()

        # 从 "计数值： X" 里解析数字，解析失败就原样写
        cnt_val = out_text
        try:
            # 兼容中文冒号/英文冒号
            s = out_text.replace("：", ":")
            if ":" in s:
                cnt_val = s.split(":", 1)[1].strip()
        except Exception:
            pass

        rows.append([in_name, cnt_val, it.get("ts", ""), it.get("out_img", "")])

    excel_path = str(sess_dir / "single_history_counts.xlsx")
    try:
        df = pd.DataFrame(rows, columns=["输入文件", "计数/状态", "时间", "输出图路径"])
        df.to_excel(excel_path, index=False)
    except Exception as e:
        raise gr.Error(f"Excel导出失败：{e}")

    # 2) zip 可视化
    zip_path = str(sess_dir / "single_history_visualizations.zip")
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            added = 0
            for it in (history or []):
                out_img = it.get("out_img")
                if out_img and os.path.isfile(out_img):
                    # 只把文件名放进压缩包
                    zf.write(out_img, arcname=os.path.basename(out_img))
                    added += 1
        if added == 0:
            # 仍返回zip，但提示更明确
            raise gr.Error("历史里没有找到可打包的输出图像文件。")
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"可视化打包失败：{e}")

    return excel_path, zip_path


USER_CREDENTIALS = {"admin": "654321"}

def check_login(username, password):
    """验证登录信息"""
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        return True, "登录成功！"
    else:
        return False, "用户名或密码错误！"

def gradio_demo():

    candidate_example_files = [
        "visualizations_cache/test_samples/2025.1.5_2025.1.5-1_17-2.jpg",
        "visualizations_cache/test_samples/2025.1.5_2025.1.5-1_24S3000-1-3.jpg",
        "visualizations_cache/test_samples/2025.1.5_2025.1.5-3_17-3.jpg",
        "visualizations_cache/test_samples/24S2982-1-1.jpg",
        "visualizations_cache/test_samples/LZX212-3.jpg"
    ]
    single_examples = [[p] for p in candidate_example_files if os.path.isfile(p)]

    candidate_zip_files = [
        "visualizations_cache/test_samples/test.zip",
    ]
    zip_examples = [[p] for p in candidate_zip_files if os.path.isfile(p)]
    watermark_css = """
            #login_section, #main_section {
                position: relative;
            }
            .gradio-container {
                background-image: none !important;
            }

            /* login/main 共用水印样式 */
            #login_section .watermark,
            #main_section .watermark {
                position: absolute;
                top: 0;
                right: 0;
                height: calc(2.6em + 1.2em + 12px);  /* 约等于两行Markdown标题高度 + 间距 */
                width: auto;
                z-index: 5;
                pointer-events: none;
                display: flex;
                align-items: flex-start;
                justify-content: flex-end;
            }

            #login_section .watermark img,
            #main_section .watermark img {
                height: 100%;
                width: auto;
                object-fit: contain;
                display: block;
            }
        """
    with gr.Blocks(title="大豆胞囊虫计数", css=watermark_css) as demo:
        logo_abs = os.path.abspath("visualizations_cache/logos/logo.png")
        session_id_state = gr.State(value=None)
        session_dir_state = gr.State(value=None)
        single_history_state = gr.State(value=[])

        
        def _init_session():
            sid, sdir = _new_session()
            _clear_session_dir(sdir)
            os.makedirs(sdir, exist_ok=True)
            return sid, sdir, [], [], []
        
        with gr.Column(visible=True, elem_id="login_section") as login_section:
            # 在登录区内放一个绝对定位的水印（高度由CSS控制为“两行Markdown高度”）
            gr.HTML(f'<div class="watermark"><img src="file={logo_abs}" /></div>')
            gr.Markdown("# 🔐 大豆胞囊虫计数系统")
            gr.Markdown("### 请先登录以使用系统")
            
            with gr.Row():
                username = gr.Textbox(
                    label="用户名",
                    value="admin",  # 默认用户名
                    placeholder="输入用户名",
                    scale=2
                )
            with gr.Row():
                password = gr.Textbox(
                    label="密码",
                    type="password",
                    value="",  # 默认密码
                    placeholder="输入密码",
                    scale=2
                )

            with gr.Row():
                login_btn = gr.Button("登录", variant="primary", size="lg")
                clear_btn = gr.Button("清除", size="lg")

            login_status = gr.Textbox(label="登录状态", visible=False)

        # 主应用界面（初始隐藏）
        with gr.Column(visible=False, elem_id="main_section") as main_section:
            gr.HTML(f'<div class="watermark"><img src="file={logo_abs}" /></div>')
            gr.Markdown("# 🌱 大豆胞囊虫计数")
            gr.Markdown("### 请选择单图推理或批量处理数据")
            

            with gr.Tab("单图精细化点回归计数"):
                with gr.Row():
                    with gr.Column(scale=1):
                        in_img = gr.Image(
                            type="filepath",
                            label="上传图片",
                            height=None,
                            width=None
                        )
                    with gr.Column(scale=1):
                        out_img = gr.Image(
                            type="filepath",
                            label="预测结果",
                            height=None,
                            width=None
                        )
                with gr.Row():
                    clear_btn_main = gr.Button("清除")
                    submit_btn = gr.Button("提交", variant="primary")

                with gr.Row():
                    out_txt = gr.Textbox(label="统计信息", lines=1)

                # 一键例子（点一下自动把例图填入输入框）
                if single_examples:
                    gr.Examples(
                        examples=single_examples,
                        inputs=[in_img],
                        label="单图测试用例"
                    )
                single_history_gallery = gr.Gallery(
                    label="历史预测输出（点击回填）",
                    columns=5,
                    height=300, 
                    show_label=True,
                    allow_preview=True,
                    object_fit="contain", 
                    )
                single_history_df = gr.Dataframe(
                    headers=["序号", "时间", "输入文件", "计数输出"],
                    value=[],
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(4, "fixed"),
                    label="历史预测（点击回填）",
                    wrap=True,
                    height=280
                    )

                
                demo.load(
                        _init_session,
                        inputs=None,
                        outputs=[
                            session_id_state,
                            session_dir_state,
                            single_history_state,
                            single_history_df,
                            single_history_gallery,
                        ],
                    )
                single_history_df.select(
                    fn=_on_history_select,
                    inputs=[single_history_state],
                    outputs=[in_img, out_img, out_txt],
                    )
                single_history_gallery.select(
                    fn=_on_history_gallery_select,
                    inputs=[single_history_state],
                    outputs=[in_img, out_img, out_txt],
                    )

                
                submit_btn.click(
                    fn=predict,
                    inputs=[in_img, session_dir_state, single_history_state],
                    outputs=[out_img, out_txt, single_history_state, single_history_df, single_history_gallery],
                    )
                clear_btn_main.click(
                    fn=lambda: [None, None, [], [], []],
                    inputs=None,
                    outputs=[in_img, out_img, single_history_state, single_history_df, single_history_gallery],
                    )
                with gr.Row():
                    export_single_btn = gr.Button("导出历史记录", variant="primary")

                with gr.Row():
                    single_out_excel = gr.File(label="历史记录报表")
                    single_out_viszip = gr.File(label="历史记录可视化")

            with gr.Tab("高通量批量图像分析"):
                zip_in = gr.File(label="上传 .zip 压缩包文件", file_types=[".zip"])
                batch_btn = gr.Button("开始批量计数", variant="primary")

                with gr.Row():
                    out_excel = gr.File(label="导出计数报表")
                    out_viszip = gr.File(label="下载所有可视化")

                out_table = gr.Dataframe(headers=["文件名", "计数/状态"], label="结果", wrap=True)

                # 一键例子（点一下自动把zip填入输入框）
                if zip_examples:
                    gr.Examples(
                        examples=zip_examples,
                        inputs=[zip_in],
                        label="批量测试用例"
                    )

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
                gr.update(value=""),
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
        
        export_single_btn.click(
            fn=export_single_history,
            inputs=[session_dir_state, single_history_state],
            outputs=[single_out_excel, single_out_viszip],
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
        debug=True,
        # allowed_paths=["visualizations_cache"]
    )

if __name__ == "__main__":
    
    initialize_model(
        cfg_path=CFG, 
        checkpoint_path=CKPT
    )
    gradio_demo()
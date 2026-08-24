# O-PET: Point-Based Object Counting with PET

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Paper](https://img.shields.io/badge/ICCV%202023-PET-6f42c1)](https://arxiv.org/abs/2308.13814)

O-PET is a research-oriented implementation of point-based object counting and localization built on [PET (Point quEry Transformer)](https://github.com/cxliu0/PET). It extends PET beyond crowd counting to configurable dense-object scenarios such as people, ships, vehicles, crops, and transit scenes.

The repository keeps PET's adaptive sparse-to-dense point-query quadtree and adds experimental components for box-agent queries, attention-based region splitting, feature enhancement, alternative matching strategies, and additional dataset loaders.

> [!IMPORTANT]
> This is research code. Dataset paths, pretrained weights, and GPU settings are environment-specific, and pretrained O-PET checkpoints are not included in this repository.

## Highlights

- **Counting and localization together** — predicts object counts and point locations from point queries.
- **Adaptive computation** — uses a quadtree splitter to route sparse and dense image regions to different query resolutions.
- **Configurable query design** — supports the original PET decoder and experimental box-agent query refinement.
- **Enhanced feature extraction** — includes VGG16-BN and Swin backbones, feature pyramids, spatial fusion, DySample upsampling, and adaptive rotated convolution options.
- **Multiple matching and loss options** — includes Hungarian or stable matching and optional probability-map supervision.
- **Broad dataset support** — loaders are provided for crowd counting, remote-sensing objects, agricultural objects, and metro scenes.
- **Distributed training** — launches with `torchrun` for single- or multi-GPU experiments.

## Repository Layout

```text
Opet/
├── configs_con/              # YAML experiment configurations
│   ├── Crowd/                # Crowd-counting experiments
│   ├── TGRS/                 # People, ship, and car experiments
│   └── Others/               # Baselines and exploratory settings
├── datasets/                 # Dataset registry, loaders, and augmentation
├── models/                   # PET, matchers, backbones, and transformers
├── notebook/                 # Log analysis and visualization notebooks
├── util/                     # Distributed, metric, logging, and FLOP utilities
├── main.py                   # Training entry point
├── eval_rebuild.py           # Evaluation and visualization entry point
├── train.sh                  # Example distributed training launcher
└── eval.sh                   # Example evaluation launcher
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/muuuuuuushroom/Opet.git
cd Opet
```

### 2. Create an isolated environment

```bash
conda create -n opet python=3.10 -y
conda activate opet
python -m pip install --upgrade pip
```

### 3. Install dependencies

Install a PyTorch build compatible with your CUDA driver by following the [official PyTorch installation guide](https://pytorch.org/get-started/locally/). Then install the remaining packages:

```bash
pip install numpy scipy matplotlib opencv-python pillow pyyaml h5py \
  scikit-learn einops timm prodigyopt
pip install "flash-linear-attention[cuda]"
```

The Gradio demo in `publish.py` additionally requires:

```bash
pip install gradio pandas
```

> [!NOTE]
> The repository does not currently provide a locked environment file. `flash-linear-attention` is imported by the transformer implementation and must match the installed PyTorch/CUDA stack. See its [installation guide](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) if dependency resolution fails.

## Data Preparation

Dataset roots are registered in [`datasets/__init__.py`](datasets/__init__.py). Update the `data_path` mapping there before running an experiment; the `data_path` value inside a YAML file is overwritten by this registry.

Supported dataset identifiers include:

| Group | `dataset_file` values | Loader / annotation style |
| --- | --- | --- |
| Crowd counting | `SHA`, `SHB`, `UCF`, `JHU`, `NWPU` | ShanghaiTech `.mat` annotations or dataset-specific point annotations |
| Remote-sensing objects | `People`, `Ship`, `Car` | Image folders with XML point/box annotations |
| Agriculture | `RTC`, `CORN`, `SOY`, `SOY_evon` | Dataset-specific list, JSON, or NumPy inputs |
| Transit | `WuhanMetro` | Train/test lists with JSON annotations |

For example, ShanghaiTech Part A is expected to follow this structure:

```text
data/Crowd_Counting/ShanghaiTech/part_A_final/
├── train_data/
│   ├── images/
│   └── ground_truth/
└── test_data/
    ├── images/
    └── ground_truth/
```

The `People`, `Ship`, and `Car` loader expects the following layout beneath each registered dataset root:

```text
<dataset-root>/
├── train_data/
│   ├── images/
│   └── VGG_anotation_truth/
└── test_data/
    ├── images/
    └── VGG_anotation_truth/
```

Other datasets have specialized layouts. Refer to the corresponding loader in `datasets/` when preparing custom data.

> [!TIP]
> Ground-truth points are represented internally as `(y, x)`. Convert annotations carefully when adding a new loader.

## Pretrained Backbones

- **Swin:** torchvision downloads ImageNet weights automatically when a Swin backbone is initialized.
- **VGG16-BN:** update `model_paths` in [`models/backbones/vgg.py`](models/backbones/vgg.py) so `vgg16_bn` points to your local ImageNet checkpoint. The current path is machine-specific.

## Training

All experiment settings are loaded from YAML. Start with one of the included configurations and adapt the dataset path, batch size, and model switches to your environment.

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  main.py \
  --cfg configs_con/TGRS/Ship.yaml
```

### Multiple GPUs

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc_per_node=2 \
  main.py \
  --cfg configs_con/TGRS/Ship.yaml
```

You can also edit the defaults at the top of `train.sh` and run:

```bash
bash train.sh
```

Resume an interrupted run with:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --standalone \
  --nproc_per_node=1 \
  main.py \
  --cfg path/to/config.yaml \
  --resume path/to/checkpoint.pth
```

Depending on the dataset, outputs are written below `outputs/<dataset>/`, `outputs/RS/<dataset>/`, or `outputs/true_cc/<dataset>/`. A run may contain:

```text
config.yaml
run_log.txt
best_checkpoint.pth
best_r2_checkpoint.pth
epoch_<N>.pth
```

## Evaluation

Evaluate a checkpoint and optionally save prediction visualizations:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_rebuild.py \
  --world_size 1 \
  --cfg path/to/config.yaml \
  --resume path/to/best_checkpoint.pth \
  --vis_dir path/to/visualizations
```

Pass an empty visualization directory to disable image output:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_rebuild.py \
  --cfg path/to/config.yaml \
  --resume path/to/best_checkpoint.pth \
  --vis_dir ""
```

The evaluator reports counting metrics such as MAE, MSE, and R², together with relative counting accuracy and localization precision where supported by the dataset.

## Configuration Guide

The most relevant YAML options are:

| Option | Purpose |
| --- | --- |
| `dataset_file` | Selects the dataset loader and output namespace |
| `backbone` | Backbone name, such as `vgg16_bn` or `swin_t` |
| `patch_size` | Training crop size |
| `context_patch` | Context region used by the quadtree splitter |
| `sparse_stride`, `dense_stride` | Query strides for sparse and dense branches |
| `attn_splitter` | Enables attention-based sparse/dense region splitting |
| `opt_query_decoder` | Enables experimental box-agent query refinement |
| `matcher` | Selects `hun` (Hungarian) or `stable` matching |
| `prob_map_lc` | Enables optional probability-map supervision with `f4x` |
| `one_key_hfy` | Enables the bundled enhanced-backbone settings |
| `one_key_zlt` | Enables the bundled query/splitter/probability-loss settings |
| `output_dir` | Experiment name below the dataset output directory |

The files in `configs_con/TGRS/` are the best starting points for people, ship, and car experiments. Use `configs_con/Crowd/SHA.yaml` for ShanghaiTech Part A.

## Custom Datasets

To add a dataset:

1. Implement a `torch.utils.data.Dataset` that returns `(image, target, auxiliary)`.
2. Store point annotations in `target["points"]` using `(y, x)` order and class labels in `target["labels"]`.
3. Add the dataset root to `data_path` in `datasets/__init__.py`.
4. Register the loader in `build_dataset`.
5. Copy an existing YAML config and update `dataset_file` and dataset-specific settings.

## Citation

This repository is derived from PET. If it is useful in your research, please cite the original ICCV 2023 paper:

```bibtex
@InProceedings{liu2023pet,
  title     = {Point-Query Quadtree for Crowd Counting, Localization, and More},
  author    = {Liu, Chengxin and Lu, Hao and Cao, Zhiguo and Liu, Tongliang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

## Acknowledgements

O-PET builds on the official [PET implementation](https://github.com/cxliu0/PET). PET, in turn, acknowledges the open-source contributions of [DETR](https://github.com/facebookresearch/detr) and [P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet).

## License

This repository does not currently include a license file. Review the upstream PET repository's academic-use notice and contact the repository maintainers before redistribution or commercial use.

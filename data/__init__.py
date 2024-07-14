#to make this folder as a module
import os
from pathlib import Path
import platform
import re
from types import SimpleNamespace

import yaml
DEFAULT_CFG_KEYS = ['task', 'mode', 'model', 'data', 'epochs', 'time', 'patience', 'batch', 'imgsz', 'save', 'save_period', 'cache', 'device', 'workers', 'project', 'name', 'exist_ok', 'pretrained', 'optimizer', 'verbose', 'seed', 'deterministic', 'single_cls', 'rect', 'cos_lr', 'close_mosaic', 'resume', 'amp', 'fraction', 'profile', 'freeze', 'multi_scale', 'overlap_mask', 'mask_ratio', 'dropout', 'val', 'split', 'save_json', 'save_hybrid', 'conf', 'iou', 'max_det', 'half', 'dnn', 'plots', 'source', 'vid_stride', 'stream_buffer', 'visualize', 'augment', 'agnostic_nms', 'classes', 'retina_masks', 'embed', 'done_warmup', 'show', 'save_frames', 'save_txt', 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'show_boxes', 'line_width', 'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs', 'warmup_momentum', 'warmup_bias_lr', 'box', 'cls', 'dfl', 'pose', 'kobj', 'label_smoothing', 'nbs', 'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud', 'fliplr', 'bgr', 'mosaic', 'mixup', 'copy_paste', 'auto_augment', 'erasing', 'crop_fraction', 'cfg']
VID_FORMATS = {"asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"}  # video suffixes
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
PIN_MEMORY = str(os.getenv("PIN_MEMORY", True)).lower() == "true"  # global pin_memory for dataloaders

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1)) 
RANK = int(os.getenv("RANK", -1))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLO
def yaml_save(file="data.yaml", data=None, header=""):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict): Data to save in YAML format.
        header (str, optional): YAML header to add.

    Returns:
        (None): Data is saved to the specified file.
    """
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings
    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    # Dump data to file in YAML format
    with open(file, "w", errors="ignore", encoding="utf-8") as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
def yaml_load(file="data.yaml", append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    assert Path(file).suffix in {".yaml", ".yml"}, f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s) or {}  # always return a dict (yaml.safe_load() may return None for empty files)
        if append_filename:
            data["yaml_file"] = str(file)
        return data
DEFAULT_CFG_DICT={'task': 'detect', 'mode': 'train', 'model': 'yolov8.yaml', 'data': 'VOC.yaml', 'epochs': 1, 'time': None, 'patience': 100, 'batch': 16, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': None, 'workers': 8, 'project': None, 'name': None, 'exist_ok': False, 'pretrained': True, 'optimizer': 'auto', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'amp': True, 'fraction': 1.0, 'profile': True, 'freeze': 'None', 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': False, 'source': 'assets', 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'done_warmup': False, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'auto_augment': 'randaugment', 'erasing': 0.4, 'crop_fraction': 1.0, 'cfg': None}
DEFAULT_CFG=SimpleNamespace(**DEFAULT_CFG_DICT)
#specific to device
RUNS_DIR=r"runs"
DATASETS_DIR=r"C:\Users\thata\intern\code\pre-built-models\datasets"
ASSETS=Path(r"C:\Users\thata\intern\code\pre-built-models\modified\assets")

weights_dir=Path("weights")
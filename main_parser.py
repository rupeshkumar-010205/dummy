import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training Configuration')
    # Basic settings
    parser.add_argument('--task',default="detect", type=str, help='YOLO task, i.e. detect, segment, classify, pose')
    parser.add_argument('--mode',default="predict" ,type=str, help='YOLO mode, i.e. train, val, predict, export, track, benchmark')
    # Train settings
    parser.add_argument('--model', type=str, default='yolov8.yaml', help='Path to model file')
    parser.add_argument('--data', type=str, default='VOC.yaml', help='Path to data file')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for')
    parser.add_argument('--time', type=float, help='Number of hours to train for (overrides epochs)')
    parser.add_argument('--patience', type=int, default=100, help='Epochs to wait for no improvement for early stopping')
    parser.add_argument('--batch', type=int, default=16, help='Number of images per batch (-1 for AutoBatch)')
    parser.add_argument('--imgsz', type=int, nargs='+', default=640, help='Input image size as int or list [w,h]')
    parser.add_argument('--save', type=bool, default=True, help='Save train checkpoints and predict results')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--cache', type=bool, default=False, help='Use cache for data loading')
    parser.add_argument('--device', type=str, help='Device to run on, e.g., cuda:0 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for data loading')
    parser.add_argument('--project', type=str, help='Project name')
    parser.add_argument('--name', type=str, help='Experiment name')
    parser.add_argument('--exist_ok', type=bool, default=False, help='Whether to overwrite existing experiment')
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use a pretrained model or load weights')
    parser.add_argument('--optimizer', type=str, default='auto', help='Optimizer to use')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print verbose output')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--deterministic', type=bool, default=True, help='Enable deterministic mode')
    parser.add_argument('--single_cls', type=bool, default=False, help='Train multi-class data as single-class')
    parser.add_argument('--rect', type=bool, default=False, help='Rectangular training if mode=train or validation if mode=val')
    parser.add_argument('--cos_lr', type=bool, default=False, help='Use cosine learning rate scheduler')
    parser.add_argument('--close_mosaic', type=int, default=10, help='Disable mosaic augmentation for final epochs')
    parser.add_argument('--resume', type=str, default=False, help='Resume training from last checkpoint')
    parser.add_argument('--amp', type=bool, default=True, help='Automatic Mixed Precision (AMP) training')
    parser.add_argument('--fraction', type=float, default=1.0, help='Dataset fraction to train on')
    parser.add_argument('--profile', type=bool, default=False, help='Profile ONNX and TensorRT speeds during training for loggers')
    parser.add_argument('--freeze', type=int, nargs='+', help='Freeze first n layers or list of layer indices during training')
    parser.add_argument('--multi_scale', type=bool, default=False, help='Whether to use multiscale during training')
    
    # Segmentation
    parser.add_argument('--overlap_mask', type=bool, default=True, help='Masks should overlap during training (segment train only)')
    parser.add_argument('--mask_ratio', type=int, default=4, help='Mask downsample ratio (segment train only)')
    
    # Classification
    parser.add_argument('--dropout', type=float, default=0.0, help='Use dropout regularization (classify train only)')
    
    # Val/Test settings
    parser.add_argument('--val', type=bool, default=True, help='Validate/test during training')
    parser.add_argument('--split', type=str, default='val', help='Dataset split to use for validation')
    parser.add_argument('--save_json', type=bool, default=False, help='Save results to JSON file')
    parser.add_argument('--save_hybrid', type=bool, default=False, help='Save hybrid version of labels')
    parser.add_argument('--conf', type=float, help='Object confidence threshold for detection')
    parser.add_argument('--iou', type=float, default=0.7, help='Intersection over union (IoU) threshold for NMS')
    parser.add_argument('--max_det', type=int, default=300, help='Maximum number of detections per image')
    parser.add_argument('--half', type=bool, default=False, help='Use half precision (FP16)')
    parser.add_argument('--dnn', type=bool, default=False, help='Use OpenCV DNN for ONNX inference')
    parser.add_argument('--plots', type=bool, default=True, help='Save plots and images during train/val')
    
    # Predict settings
    parser.add_argument('--source', type=str, default='assets', help='Source directory for images or videos')
    parser.add_argument('--vid_stride', type=int, default=1, help='Video frame-rate stride')
    parser.add_argument('--stream_buffer', type=bool, default=False, help='Buffer all streaming frames')
    parser.add_argument('--visualize', type=bool, default=False, help='Visualize model features')
    parser.add_argument('--augment', type=bool, default=False, help='Apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', type=bool, default=False, help='Class-agnostic NMS')
    parser.add_argument('--classes', type=int, nargs='+', help='Filter results by class')
    parser.add_argument('--retina_masks', type=bool, default=False, help='Use high-resolution segmentation masks')
    parser.add_argument('--embed', type=int, nargs='+', help='Return feature vectors/embeddings from given layers')
    parser.add_argument('--done_warmup', type=bool, default=False)
    parser.add_argument('--webcam',default=False,type=bool,help="provide the link for IP cam else uses system cam")
    
    # Visualize settings
    parser.add_argument('--show', type=bool, default=False, help='Show predicted images and videos')
    parser.add_argument('--save_frames', type=bool, default=False, help='Save predicted individual video frames')
    parser.add_argument('--save_txt', type=bool, default=False, help='Save results as .txt file')
    parser.add_argument('--save_conf', type=bool, default=False, help='Save results with confidence scores')
    parser.add_argument('--save_crop', type=bool, default=False, help='Save cropped images with results')
    parser.add_argument('--show_labels', type=bool, default=True, help='Show prediction labels')
    parser.add_argument('--show_conf', type=bool, default=True, help='Show prediction confidence')
    parser.add_argument('--show_boxes', type=bool, default=True, help='Show prediction boxes')
    parser.add_argument('--line_width', type=int, help='Line width of the bounding boxes')

    # Hyperparameters
    parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='Final learning rate')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimizer weight decay')
    parser.add_argument('--warmup_epochs', type=float, default=3.0, help='Warmup epochs')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup initial momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1, help='Warmup initial bias lr')
    parser.add_argument('--box', type=float, default=7.5, help='Box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='Cls loss gain')
    parser.add_argument('--dfl', type=float, default=1.5, help='Dfl loss gain')
    parser.add_argument('--pose', type=float, default=12.0, help='Pose loss gain')
    parser.add_argument('--kobj', type=float, default=1.0, help='Keypoint obj loss gain')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing')
    parser.add_argument('--nbs', type=int, default=64, help='Nominal batch size')
    parser.add_argument('--hsv_h', type=float, default=0.015, help='Image HSV-Hue augmentation')
    parser.add_argument('--hsv_s', type=float, default=0.7, help='Image HSV-Saturation augmentation')
    parser.add_argument('--hsv_v', type=float, default=0.4, help='Image HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0, help='Image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='Image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='Image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='Image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0, help='Image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='Image flip left-right (probability)')
    parser.add_argument('--bgr', type=float, default=0.0, help='Image channel BGR (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='Image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='Image mixup (probability)')
    parser.add_argument('--copy_paste', type=float, default=0.0, help='Segment copy-paste (probability)')
    parser.add_argument('--auto_augment', type=str, default='randaugment', help='Auto augmentation policy')
    parser.add_argument('--erasing', type=float, default=0.4, help='Probability of random erasing during classification training')
    parser.add_argument('--crop_fraction', type=float, default=1.0, help='Image crop fraction for classification')

    # Custom config.yaml
    parser.add_argument('--cfg', type=str, help='Path to custom config.yaml')

    args = parser.parse_args()
    return args
if __name__=="__main__":
    args=parse_args()
    print(args)
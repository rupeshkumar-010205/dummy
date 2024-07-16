# YOLOv8 Custom Object Detection

This repository is a customized version of the Ultralytics YOLOv8 code, tailored specifically for object detection. The original YOLOv8 implementation by Ultralytics supports multiple image tasks such as segmentation, pose estimation, and oriented bounding boxes. However, this repository focuses solely on object detection, simplifying the code and making it easier to read and modify.

## Features
- **Simplified Code**: The codebase has been streamlined for easier readability and modification.
- **Focused Functionality**: Removed support for segmentation, pose estimation, and oriented bounding boxes to concentrate solely on object detection.


### Training

To train the model, use the following command:

```bash
python main.py --mode train --model yolov8.yaml --data VOC.yaml --epochs 10
```

#### Training Options

- `--model`: Path to the model configuration file (default: `yolov8.yaml`).
- `--data`: Path to the data configuration file (default: `VOC.yaml`).
- `--epochs`: Number of epochs to train for (default: 1).

Additional options can be found in `main.py`.

### Validation

To validate the model, use the following command:

```bash
python main.py --mode val --model runs\detect\train\weights\best.pt --data VOC.yaml
```

#### Validation Options

- `--model`: Path to the model configuration file (default: `yolov8.yaml`).
- `--data`: Path to the data configuration file (default: `VOC.yaml`).

Additional options can be found in `main.py`.

### Prediction

To run predictions on images, use the following command:

```bash
python main.py --mode predict --model runs/detect/train/weights/best.pt --source assets/bus.jpg
```

To use a webcam for predictions, use the following command:

```bash
python main.py --mode predict --model runs/detect/train/weights/best.pt --webcam True
```

#### Prediction Options

- `--model`: Path to the model weights file (default: `runs/detect/train/weights/best.pt`).
- `--source`: Source directory for images or videos (default: `assets`).
- `--webcam`: Use the system webcam for live predictions (default: `False`).

Additional options can be found in `main.py`.

## Available Weights

This repository contains weights trained on the following datasets:

- Pascal VOC dataset: Trained for 10 epochs

Those weights are being used as default for val and predict 

## Additional Information

For a comprehensive list of all configurable options, refer to the argument parser in `main.py`.



## Packages , versions and Licenses

| Name                | Version     | License                                                                |
|---------------------|-------------|------------------------------------------------------------------------|
| Bottleneck          | 1.3.7       | BSD License                                                            |
| Brotli              | 1.0.9       | MIT License                                                            |
| Jinja2              | 3.1.4       | BSD License                                                            |
| MarkupSafe          | 2.1.3       | BSD License                                                            |
| PySocks             | 1.7.1       | BSD                                                                    |
| PyYAML              | 6.0.1       | MIT License                                                            |
| certifi             | 2024.6.2    | Mozilla Public License 2.0 (MPL 2.0)                                   |
| charset-normalizer  | 2.0.4       | MIT License                                                            |
| colorama            | 0.4.6       | BSD License                                                            |
| contourpy           | 1.2.0       | BSD License                                                            |
| cycler              | 0.11.0      | BSD License                                                            |
| filelock            | 3.13.1      | The Unlicense (Unlicense)                                              |
| fonttools           | 4.51.0      | MIT License                                                            |
| fsspec              | 2024.6.1    | BSD License                                                            |
| gmpy2               | 2.1.2       | GNU Lesser General Public License v3 or later (LGPLv3+)                |
| idna                | 3.7         | BSD License                                                            |
| importlib-resources | 6.1.1       | Apache Software License                                                |
| intel-openmp        | 2021.4.0    | Other/Proprietary License                                              |
| kiwisolver          | 1.4.4       | BSD License                                                            |
| matplotlib          | 3.8.4       | Python Software Foundation License                                     |
| mkl                 | 2021.4.0    | Other/Proprietary License                                              |
| mkl-fft             | 1.3.1       | BSD                                                                    |
| mkl-random          | 1.2.2       | BSD                                                                    |
| mkl-service         | 2.4.0       | BSD                                                                    |
| mpmath              | 1.3.0       | BSD License                                                            |
| networkx            | 3.2.1       | BSD License                                                            |
| numexpr             | 2.8.4       | MIT License                                                            |
| numpy               | 1.24.3      | BSD License                                                            |
| opencv-python       | 4.10.0.84   | Apache Software License                                                |
| packaging           | 23.2        | Apache Software License; BSD License                                   |
| pandas              | 2.2.2       | BSD License                                                            |
| pillow              | 10.4.0      | Historical Permission Notice and Disclaimer (HPND)                     |
| psutil              | 6.0.0       | BSD License                                                            |
| py-cpuinfo          | 9.0.0       | MIT License                                                            |
| pycocotools         | 2.0.8       | FreeBSD                                                                |
| pyparsing           | 3.0.9       | MIT License                                                            |
| python-dateutil     | 2.9.0.post0 | Apache Software License; BSD License                                   |
| pytz                | 2024.1      | MIT License                                                            |
| requests            | 2.32.2      | Apache Software License                                                |
| scipy               | 1.13.1      | BSD License                                                            |
| seaborn             | 0.13.2      | BSD License                                                            |
| six                 | 1.16.0      | MIT License                                                            |
| sympy               | 1.12        | BSD License                                                            |
| tbb                 | 2021.13.0   | Other/Proprietary License                                              |
| torch               | 2.3.1       | BSD License                                                            |
| torchaudio          | 2.3.1       | BSD License                                                            |
| torchaudio          | 2.3.1       | BSD License                                                            |
| torchvision         | 0.18.1      | BSD                                                                    |
| tqdm                | 4.66.4      | MIT License; Mozilla Public License 2.0 (MPL 2.0)                      |
| typing_extensions   | 4.11.0      | Python Software Foundation License                                     |
| tzdata              | 2023.3      | Apache Software License                                                |
| unicodedata2        | 15.1.0      | Apache License 2.0                                                     |
| urllib3             | 2.2.2       | MIT License                                                            |
| win-inet-pton       | 1.1.0       | This software released into the public domain. Anyone is free to copy, |
| zipp                | 3.17.0      | MIT License                                                            |


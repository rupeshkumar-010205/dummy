import glob
import cv2
from torch import nn
import math
from matplotlib import patches, pyplot as plt
import torch
from data import DEFAULT_CFG, DEFAULT_CFG_DICT
from data import yaml_load
from data.build import build_dataloader
from data.dataset import YOLODataset, check_det_dataset
from model import get_latest_run
from model.train import DetectionTrainer
import time
from PIL import Image
import torchvision
# dataset=YOLODataset(
#     img_path=check_det_dataset("VOC.yaml")["train"],
#     data=DEFAULT_CFG_DICT
# )
# loader=build_dataloader(dataset=dataset,batch=16,workers=0)


# output=dataset[1000]
# print(output)

"""
training test
"""
# trainer=DetectionTrainer()
# start_time=time.time()
# results=trainer.predict(source=r"C:\Users\thata\intern\code\pre-built-models\modified\classroom.mp4",stream=False)
# print(time.time()-start_time)
"""
dataset test
"""
# data=check_det_dataset("VOC.yaml")
# dataset=build_yolo_dataset(DEFAULT_CFG,data["val"],batch=16,data=data,mode="val")

# print(dataset[130])


# model=torch.load(r"C:\Users\thata\intern\code\pre-built-models\modified\runs\coco\weights\best.pt")
# print(model)

# print(get_latest_run())


# ckpt=torch.load(r"C:\Users\thata\intern\code\pre-built-models\modified\runs\detect\train\weights\best.pt")

# model=ckpt.get("ema") or ckpt.get("model")
# model.float()
# inp=torch.ones((1,3,640,640)).cuda()
# print(model(inp))



# data=check_det_dataset("coco.yaml")
# dataset=YOLODataset(img_path=data["train"],data=data,hyp=DEFAULT_CFG)
# output=dataset[150]

# # Convert image tensor to numpy array
# img_np = output['img'].numpy().transpose(1, 2, 0)

# # Extract bounding box and rescale to image dimensions
# bboxes = output['bboxes'].numpy()
# img_height, img_width = output['resized_shape']

# # Rescale bounding box coordinates
# bboxes[:, 0] *= img_width  # x_center
# bboxes[:, 1] *= img_height # y_center
# bboxes[:, 2] *= img_width  # width
# bboxes[:, 3] *= img_height # height

# # Convert bounding box from center format (x_center, y_center, width, height) to (x_min, y_min, width, height)
# bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2  # x_min
# bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2  # y_min

# # Plot the image
# fig, ax = plt.subplots(1)
# ax.imshow(img_np)

# # Draw bounding boxes
# for bbox in bboxes:
#     rect = patches.Rectangle(
#         (bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none'
#     )
#     ax.add_patch(rect)

# plt.axis('off')
# plt.show()

# dataloader=build_dataloader(dataset,batch=16,workers=0,shuffle=True)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# def show_first_image_with_labels(dataloader, class_names):
#     # Retrieve the first batch
#     first_batch = next(iter(dataloader))
#     print(first_batch["img"][0])
#     print(first_batch["bboxes"][0])
#     print(first_batch["cls"][0])
#     # Load the image
#     image = first_batch["img"][0]
#     # Convert image tensor to PIL image
#     image_np = first_batch["img"][0].numpy().transpose(1, 2, 0)  # Convert to HWC format
#     image_pil = Image.fromarray(image_np)
#     # Convert bounding box from center format (x_center, y_center, width, height) to corner format (x_min, y_min, width, height)
#     x_center, y_center, width, height = first_batch["bboxes"][0]
#     img_width, img_height = image_pil.size
#     x_min = (x_center - width / 2) * img_width
#     y_min = (y_center - height / 2) * img_height
#     width = width * img_width
#     height = height * img_height

#     # Display the image with the bounding box
#     fig, ax = plt.subplots(1)
#     ax.imshow(image_pil)

#     # Create a rectangle patch
#     rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

#     plt.axis('off')
#     plt.show()
#     # # Extract the first image tensor, class labels, and bounding boxes from the batch
#     # first_image_tensor = cv2.imread(first_batch['im_file'][0])
#     # first_class_labels = first_batch['cls']
#     # first_bboxes = first_batch['bboxes']
#     # batch_idx = first_batch['batch_idx']
    
#     # # Filter bounding boxes and labels for the first image
#     # indices = batch_idx == 0
#     # bboxes = first_bboxes[indices]
#     # labels = first_class_labels[indices]
    
#     # # # Convert the tensor to a NumPy array and transpose it to (H, W, C)
#     # # first_image_np = first_image_tensor.numpy().transpose(1, 2, 0)
    
#     # # Get the corresponding image file path
#     # first_image_file = first_batch['im_file'][0]
    
#     # # Plot the image
#     # fig, ax = plt.subplots(1)
#     # ax.imshow(first_image_tensor)
    
#     # # Draw the bounding boxes and class labels
#     # for bbox, label in zip(bboxes, labels):
#     #     # Get the bounding box coordinates
#     #     x_center, y_center, width, height = bbox
#     #     x = x_center - width / 2
#     #     y = y_center - height / 2
        
#     #     # Create a rectangle patch
#     #     rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor='none')
#     #     ax.add_patch(rect)
        
#     #     # Add the class label
#     #     class_name = class_names[int(label)]
#     #     plt.text(x, y, class_name, color='white', fontsize=12, backgroundcolor='red')
    
#     # plt.title(f"First Image: {first_image_file}")
#     # plt.axis('off')
#     # plt.show()

# # Example usage
# # Assuming `dataloader` is your DataLoader object and `class_names` is a list of class names
# class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
#         7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
#         14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
#         21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie',
#         28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
#         35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup',
#         42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
#         50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 
#         56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv',
#         63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven',
#         70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
#         77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# show_first_image_with_labels(dataloader, class_names)
# ckpt=torch.load(r"C:\Users\thata\intern\code\pre-built-models\modified\runs\detect\train17\weights\best.pt")
# model=ckpt["ema"] or ckpt["model"]
# torch.save(model.state_dict(),"weights.pt")

# ckpt=torch.load(r"C:\Users\thata\intern\code\pre-built-models\modified\runs\detect\train2\weights\best.pt")

# model=ckpt["ema"] or ckpt["model"]
# print(model)

# model.load_state_dict(torch.load(r"C:\Users\thata\intern\code\pre-built-models\modified\weights.pt"))


# ckpt["model"]=model


# torch.save(ckpt,"model.pt")

# print(yaml_model_load(r"C:\Users\thata\intern\code\pre-built-models\modified\data\yolov8.yaml"))

# print(get_latest_run())
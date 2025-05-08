import os
from IPython.display import Image
from metaseg import SegAutoMaskPredictor
import torch
device = 'cuda'
# 初始化模型，只下载一次权重
predictor = SegAutoMaskPredictor()  # 选择模型类型
# 输入和输出文件夹路径
import os
from PIL import Image

# 指定要遍历的目录路径
directory = "/home/diandian/Diandian/DD/LV_DATA"

# 遍历目录下的文件和文件夹
for root, dirs, files in os.walk(directory):
    for file in files:
        # 如果是 PNG 文件
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            if 'gt' in root or 'Desplash' not in root or 'level' in root or '._' in file:
                continue
            file_path = os.path.join(root, file)
            print(file_path)
            output_root = root.replace('LV_DATA', 'LV_DATA_SAM')
            os.makedirs(output_root, exist_ok=True)
            output_path = os.path.join(output_root, file)
            results = predictor.image_predict(
                source=file_path,
                model_type="vit_h",
                points_per_side=32,
                points_per_batch=64,
                min_area=0,
                output_path=output_path,
                show=False,
                save=True,
            )
            print(f"Processed {file_path} and saved to {output_path}")





# input_folder = "/home/diandian/Diandian/DD/LV_DATA/Defog/test/input"  # 替换为你存放待分割图像的文件夹路径
# output_folder = "/home/diandian/Diandian/DD/LV_DATA/SAM/Defog/test/input"  # 替换为你希望保存分割结果的文件夹路径
#
# # 创建输出文件夹（如果不存在）
# os.makedirs(output_folder, exist_ok=True)
#
# # 遍历输入文件夹中的所有图片文件
# for filename in os.listdir(input_folder):
#     # 构造完整的文件路径
#     input_path = os.path.join(input_folder, filename)
#     output_path = os.path.join(output_folder, f"seg_{filename}")  # 保存分割结果的路径
#
#     # 检查是否是图像文件（可以根据需要添加更多扩展名）
#     if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#         # 执行图像分割
#         results = predictor.image_predict(
#             source=input_path,
#             model_type="vit_h",
#             points_per_side=32,
#             points_per_batch=64,
#             min_area=0,
#             output_path=output_path,
#             show=False,
#             save=True,
#         )
#         print(f"Processed {filename} and saved to {output_path}")
#
# # 显示其中一个分割结果
# Image(os.path.join(output_folder, os.listdir(output_folder)[0]))
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


image = cv2.imread('./761.jpg')
# 还原原图像色彩
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('on')
plt.show()
"""

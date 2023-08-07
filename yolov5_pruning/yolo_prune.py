import torch.nn.utils.prune as prune 
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from models.yolo import Model
from torchsummary import summary

# 定義模型
device = torch.device("cuda"if torch.cuda.is_available()else 'cpu')
 
cfg = '/Users/danyow/Desktop/model_pruning/yolov5_pruning/models/yolov5n.yaml'
model = Model(cfg).to(device)  # create model
'''下面是官方示範'''
# for name, m in model.named_modules():
#     if isinstance(m, nn.Conv2d): # 如果是conv就進行剪枝
#         prune.l1_unstructured(m, name='weight', amount=0.3)  # prune
#         prune.remove(m, 'weight')  # make permanent
        
'''借鑑+修改
第一層使用非結構化剪枝'''
module = None
for name, m in model.named_modules():
    # print(name)
    # print('檢驗的',m)
    # print(type(m))
    # print('-'*20)

    if isinstance(m, nn.Conv2d): # 如果是conv就進行剪枝
        module = m 
        break

prune.random_unstructured(module, name="weight", amount=0.3)
prune.remove(m, 'weight')  # make permanent 使微調有效
# prune.l1_unstructured(module,name='bias',amount=3)# 不用加這個因為沒有偏差

print('微調之後的模型，樣式')
print(model)

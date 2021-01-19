---
title: detectron understanding (0)
date: 2021-01-19 10:38:27
tags: Deep Learning Network
---

```shell
- tools
    - train_net.py: 训练
    - test_net.py: 测试
- configs
    - 各种网络的配置文件.yml
- lib
    - core：
        - config.py: 定义了通用型rcnn可能用到的所有超参
        - test_engine.py: 整个测试流程的控制器
        - test.py
    - dataset: 原始数据IO、预处理
        - your_data.py：在这里定义你自己的数据、标注读取方式
        - roidb.py
    - roi_data：数据工厂，根据config的网络配置生成需要的各种roi、anchor等
        - loader.py
        - rpn.py: 生成RPN需要的blob
        - data_utils.py: 生成anchor
    - modeling: 各种网络插件，rpn、mask、fpn等
        - model_builder.py：构造generalized rcnn
        - ResNet.py: Resnet backbone相关
        - FPN.py：RPN with an FPN backbone
        - rpn_heads.py：RPN and Faster R-CNN outputs and losses
        - mask_rcnn_heads：Mask R-CNN outputs and losses
    - utils：小工具
```

“detectron 1” hierarchical structure lists as above. 

## dataset prepare

If we need to prepare custom datasets, cd to ``lib/datasets`` or ``data/datasets(maskrcnn repo)`` folder, we need to construct one ```custom_name.py``` and add that to ```__init__.py``` like below:

```python
# __init__.py

from .coco import COCODataset
from .voc import PascalVOCDataset
from .concat_dataset import ConcatDataset
from .visual_genome import VGDataset
from .custom_dataset import CustomeDataset

__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "VGDataset", "CustomDataset"]
```

```python
# custom_dataset.py

import torch
import torchvision

from maskrcnn_benchmark.structures.bounding_box import BoxList

class CustomDataset(torch.utils.data.Dataset):
	CLASSES: (
		"__background__", "label_1", "label_2", ... ,
	)
	
	def __init__(self, data_dir, split, ...):
	def __getitem__(self, index):
	...
```




# numpy & friends
import cv2
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# transformers & friends
import segmentation_models_pytorch as smp
from einops import rearrange, repeat, reduce
from timm.models.layers import drop_path, trunc_normal_
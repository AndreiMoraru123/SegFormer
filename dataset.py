# standard library imports
import os
from collections import namedtuple

# numpy and friends
import cv2
import numpy as np

# pytorch
import torch
from torchvision import transforms
from torch.utils.data import Dataset

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.56, 0.406), std=(0.229, 0.224, 0.225))
])

cs_labels = namedtuple('CityscapesClass', ['name', 'train_id', 'color'])
cs_classes = [
    cs_labels('road', 0, (128, 64, 128)),
    cs_labels('sidewalk', 1, (244, 35, 232)),
    cs_labels('building', 2, (70, 70, 70)),
    cs_labels('wall', 3, (102, 102, 156)),
    cs_labels('fence', 4, (190, 153, 153)),
    cs_labels('pole', 5, (153, 153, 153)),
    cs_labels('traffic light', 6, (250, 170, 30)),
    cs_labels('traffic sign', 7, (220, 220, 0)),
    cs_labels('vegetation', 8, (107, 142, 35)),
    cs_labels('terrain', 9, (152, 251, 152)),
    cs_labels('sky', 10, (70, 130, 180)),
    cs_labels('person', 11, (220, 20, 60)),
    cs_labels('rider', 12, (255, 0, 0)),
    cs_labels('car', 13, (0, 0, 142)),
    cs_labels('truck', 14, (0, 0, 70)),
    cs_labels('bus', 15, (0, 60, 100)),
    cs_labels('train', 16, (0, 80, 100)),
    cs_labels('motorcycle', 17, (0, 0, 230)),
    cs_labels('bicycle', 18, (119, 11, 32)),
    cs_labels('ignore_class', 19, (0, 0, 0)),
]

train_id_to_color = np.array([c.color for c in cs_classes
                              if (c.train_id != -1
                                  and c.train_id != 255)
                              ])


class CityscapesDataset(Dataset):
    """
    Dataset class for Cityscapes semantic segmentation data
    """

    def __init__(self, rootDir: str, folder: str, tf: transforms = None):
        self.rootDir = rootDir
        self.folder = folder
        self.tf = tf

        sourceFolder = os.path.join(rootDir, 'leftImg8bit', folder)
        self.imgPaths = [os.path.join(sourceFolder, f)
                         for f in sorted(os.listdir(sourceFolder))
                         if f.endswith('.png')]

        labelFolder = os.path.join(rootDir, 'gtFine', folder)
        self.labelPaths = [os.path.join(labelFolder, f)
                           for f in sorted(os.listdir(labelFolder))
                           if f.endswith('.png')]

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, idx):
        sourceImg = cv2.imread(self.imgPaths[idx], flags=cv2.IMREAD_UNCHANGED)
        sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB)

        if self.tf is not None:
            sourceImg = self.tf(sourceImg)

        labelImg = cv2.imread(self.labelPaths[idx], flags=cv2.IMREAD_UNCHANGED)
        labelImg[labelImg == 255] = 19
        labelImg = torch.from_numpy(labelImg).long()

        return sourceImg, labelImg


# cs = CityscapesDataset(rootDir=r'D:\SemanticSegmentation', folder='train')
#
# img_1, label_1 = cs[0]
#
# cv2.imshow('img_1', img_1)
# cv2.waitKey(0)
#
# label_1 = train_id_to_color[label_1.numpy()].astype(np.uint8)
# label_1 = cv2.cvtColor(label_1, cv2.COLOR_RGB2BGR)
#
# cv2.imshow('label_1', label_1)
# cv2.waitKey(0)

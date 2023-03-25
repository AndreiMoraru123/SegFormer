""" Test the dataset module """

import pytest
import os

# PyTorch
import torch

# Model
from model import SegFormer


current_dir = os.path.dirname(os.path.abspath(__file__))
weights_file = os.path.join(current_dir, '..', 'weights', 'segformer_mit_b3_imagenet_weights.pt')


@pytest.fixture(scope="module")
def model():
    """ Returns a SegFormer instance """
    return SegFormer(in_channels=3, num_classes=19)


def test_backbone(model):
    """ Test the SegFormer class by loading the MIT pretrained weights """
    model.backbone.load_state_dict(torch.load(weights_file))

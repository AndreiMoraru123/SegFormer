"""Test the dataset module."""
import pytest

# PyTorch
import torch

# Dataset
from dataset import CityscapesDataset, get_data_splits


@pytest.fixture(scope="module")
def dataset():
    """Returns a CityscapesDataset instance."""
    return CityscapesDataset(rootDir=r'D:\SemanticSegmentation', folder='train', tf=None)


def test_dataset(dataset):
    """Test the CityscapesDataset class."""
    assert len(dataset) == 2975
    assert dataset[0][0].shape == (512, 1024, 3)  # RGB image
    assert dataset[0][1].shape == (512, 1024)  # label
    assert dataset[0][1].dtype == torch.int64  # label is integer
    assert dataset[0][1].max() == 19  # background class is 19
    assert dataset[0][1].min() == 0  # road class is 0


def test_dataset_splits(dataset):
    """Test the get_data_splits function."""
    train_set, dev_set, test_set = get_data_splits(rootDir=r'D:\SemanticSegmentation')
    assert len(train_set) == int(0.8 * len(dataset)) == 0.8 * 2975 == 2380
    assert len(dev_set) == len(dataset) - len(train_set) == 2975 - 2380 == 595
    assert len(test_set) == 500
    assert train_set[0][0].shape == (3, 512, 1024)  # PyTorch moves channel dimension to first
    assert train_set[0][1].shape == (512, 1024)  # label
    assert train_set[0][1].dtype == torch.int64  # label is integer
    assert train_set[0][1].max() == 19  # background class is 19
    assert train_set[0][1].min() == 0  # road class is 0


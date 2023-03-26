""" Train the model on the Cityscapes dataset """

# PyTorch
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

# Transformers & friends
import segmentation_models_pytorch as smp

# Utils
from utils import meanIoU
from utils import get_dataloaders
from utils import train_validate_model

# Dataset
from dataset import get_data_splits

# Model
from model import SegFormer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on: ", torch.cuda.get_device_name(0))

targetWidth = 1024
targetHeight = 512

N_EPOCHS = 50
NUM_CLASSES = 19
MAX_LR = 1e-3
MODEL_NAME = f'SegFormer_{targetWidth}x{targetHeight}_epochs{N_EPOCHS}_lr{MAX_LR}'
output_path = 'trained_models'

criterion = smp.losses.FocalLoss('multiclass', ignore_index=19)

model = SegFormer(in_channels=3, num_classes=NUM_CLASSES).to(device)
model.backbone.load_state_dict(torch.load('weights/segformer_mit_b3_imagenet_weights.pt', map_location=device))

optimizer = optim.Adam(model.parameters(), lr=MAX_LR, betas=(0.9, 0.999),
                       eps=1e-08, weight_decay=1e-4, amsgrad=True)


train_set, dev_set, test_set = get_data_splits(rootDir=r'D:\SemanticSegmentation')
train_loader, dev_loader, test_loader = get_dataloaders(train_set, dev_set, test_set)

# sample_image, sample_label = train_set[0]
# print(f"There are {len(train_set)} train images, {len(dev_set)} validation images, {len(test_set)} test Images")
# print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

scheduler = OneCycleLR(optimizer, max_lr=MAX_LR, epochs=N_EPOCHS, steps_per_epoch=len(train_loader),
                       pct_start=0.1, anneal_strategy='linear', final_div_factor=10)


if __name__ == '__main__':

    train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer,
                         device, train_loader, dev_loader, meanIoU, 'meanIoU',
                         NUM_CLASSES, scheduler, output_path)
from utils import *
from model import SegFormer
from train import test_loader, test_set, criterion, meanIoU, NUM_CLASSES
from dataset import train_id_to_color

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_path = 'trained_models'
MODEL_NAME = 'SegFormer_1024x512_epochs50_lr0.001'

model = SegFormer(in_channels=3, num_classes=NUM_CLASSES).to(device)

if __name__ == '__main__':

    model.load_state_dict(torch.load(f'{output_path}/{MODEL_NAME}.pt', map_location=device))
    _, test_metric = evaluate_model(model, test_loader, criterion, meanIoU, NUM_CLASSES, device)
    print(f"\nModel has {test_metric} mean IoU in test set")

    num_test_samples = 2
    _, axes = plt.subplots(num_test_samples, 3, figsize=(3 * 6, num_test_samples * 4))
    visualize_predictions(model, test_set, axes, device, numTestSamples=num_test_samples,
                          id_to_color=train_id_to_color)
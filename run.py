from utils import *
from model import SegFormer
from train import test_loader, test_set, criterion, meanIoU, NUM_CLASSES
from dataset import train_id_to_color

import os
from tqdm import tqdm
from utils import preprocess


def predict_cs_video(model, model_name, demo_video_path, id_to_color,
                     output_dir, target_width, target_height, device,
                     fps: int = 20, alpha: float = 0.3):
    test_images = [os.path.join(demo_video_path, *[x]) for x in sorted(os.listdir(demo_video_path))]

    output_filename = f'{model_name}_cs_part_overlay_demo_video.mp4'
    output_video_path = os.path.join(output_dir, *[output_filename])

    # handles for input output videos
    output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                    fps, (target_width, target_height))

    # create progress bar
    num_frames = int(len(test_images))
    pbar = tqdm(total=num_frames, position=0, leave=True)

    for i in range(num_frames):
        frame = cv2.imread(test_images[i])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # create torch tensor to give as input to model
        pt_image = preprocess(frame)
        pt_image = pt_image.to(device)

        # get model prediction and remap certain labels to showcase
        # only certain colors. class index 19 has color map (0,0,0),
        # so remap unwanted classes to 19
        y_pred = torch.argmax(model(pt_image.unsqueeze(0)), dim=1).squeeze(0)
        predicted_labels = y_pred.cpu().detach().numpy()

        # convert to corresponding color
        cm_labels = (id_to_color[predicted_labels]).astype(np.uint8)

        # overlay prediction over input frame
        overlay_image = cv2.addWeighted(frame, 1, cm_labels, alpha, 0)
        overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR)

        # write output result and update progress
        output_handle.write(overlay_image)
        pbar.update(1)

    output_handle.release()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_NAME = 'SegFormer_1024x512_epochs50_lr0.001'
targetWidth = 1024
targetHeight = 512
model = SegFormer(in_channels=3, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(f'trained_models/{MODEL_NAME}.pt', map_location=device))


if __name__ == '__main__':
    predict_cs_video(model, MODEL_NAME,
                     demo_video_path='demoVideo/custom',
                     id_to_color=train_id_to_color,
                     output_dir='demo',
                     target_width=targetWidth,
                     target_height=targetHeight,
                     device=device,
                     alpha=0.7)

from utils import *
from model import SegFormer

import os
from tqdm import tqdm

import torch.nn.functional as F


def preprocess_image(image_path, tf, patch_size):
    # read image -> convert to RGB -> torch Tensor
    rgb_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img = tf(rgb_img)
    _, image_height, image_width = img.shape

    # make the image divisible by the patch size
    w, h = image_width - image_width % patch_size, image_height - image_height % patch_size
    img = img[:, :h, :w].unsqueeze(0)

    w_featmap = img.shape[-1] // patch_size
    h_featmap = img.shape[-2] // patch_size
    return rgb_img, img, w_featmap, h_featmap


def calculate_attentions(img, w_featmap, h_featmap, patch_size, mode='bilinear'):
    attentions = model.get_last_self_attn(img.to(device))
    nh = attentions.shape[1]

    # we keep only the output patch attention
    # reshape to image size
    attentions = attentions[0, :, :, 0].reshape(nh, h_featmap, w_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode=mode)[0].detach().cpu().numpy()
    return attentions


def get_attention_masks(image_path, model, transform, patch_size, mode='bilinear'):
    rgb_img, img, w_featmap, h_featmap = preprocess_image(image_path, transform, patch_size)
    attentions = calculate_attentions(img, w_featmap, h_featmap, patch_size, mode=mode)
    return rgb_img, attentions


def convert_images_to_video(images_dir, output_video_path, targetWidth, targetHeight, fps: int = 20):
    input_images = [os.path.join(images_dir, *[x]) for x in sorted(os.listdir(images_dir))]

    if len(input_images) > 0:
        sample_image = cv2.imread(input_images[0])
        # height, width, _ = sample_image.shape

        # handles for input output videos
        output_handle = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                        fps, (targetWidth, targetHeight))

        # create progress bar
        num_frames = int(len(input_images))
        pbar = tqdm(total=num_frames, position=0, leave=True)

        for i in tqdm(range(num_frames), position=0, leave=True):
            frame = cv2.imread(input_images[i])
            output_handle.write(frame)
            pbar.update(1)

        # release the output video handler
        output_handle.release()

    else:
        pass


def createDir(dirPath):
    if not os.path.isdir(dirPath):
        os.mkdir(dirPath)


output_dir = '.'
patch_size = 32
stage_scale = [4, 8, 16, 32]
stage_heads = [1, 2, 5, 8]
titles = []
for stage_index, stage_nh in enumerate(stage_heads):
    titles.extend([f"STAGE_{stage_index + 1}_HEAD_{x + 1}" for x in range(stage_nh)])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

targetWidth = 1024
targetHeight = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 19
MODEL_NAME = 'SegFormer_1024x512_epochs50_lr0.001'
output_path = 'dataset'

model = SegFormer(in_channels=3, num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(f'trained_models/{MODEL_NAME}.pt', map_location=device))
model.eval()

if __name__ == "__main__":

    # font = {'family': 'normal', 'weight': 'bold', 'size': 4}
    # plt.rc('font', **font)
    plt.rcParams['text.color'] = 'white'

    input_dir = 'demoVideo/stuttgart_00'
    image_list = sorted(os.listdir(input_dir))
    images_path = [os.path.join(input_dir, x) for x in image_list]

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 8))
    axes = axes.flatten()
    fig.tight_layout()

    image_folder = 'attention_images'

    for image_path in tqdm(images_path):
        image_name = image_path.split(os.sep)[-1].split('.')[0]

        rgb_img, attentions = get_attention_masks(image_path, model, transform, patch_size, mode='bilinear')

        for i in range(len(axes)):
            axes[i].clear()
            if (i < 4):
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i + 8], x=0.20, y=0.9, va="top")

            elif i == 4:
                axes[i].imshow(np.zeros_like(rgb_img))
            else:
                axes[i].imshow(rgb_img)
                axes[i].imshow(attentions[i - 1], cmap='inferno', alpha=0.5)
                axes[i].set_title(titles[i - 1 + 8], x=0.20, y=0.9, va="top")

            axes[i].axis('off')

        fig.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(f'{image_folder}/{image_name}_last_stage.png')

    video_output_dir = os.path.join(output_dir, *['attention_videos'])
    createDir(video_output_dir)
    output_video_path = os.path.join(video_output_dir, *[f"{MODEL_NAME}_last_stage_demoVideo.mp4"])

    attention_dir = "attention_images"

    convert_images_to_video(attention_dir, output_video_path, targetWidth, targetHeight)

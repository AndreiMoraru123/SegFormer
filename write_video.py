import cv2
import os
from tqdm import tqdm


def extract_frames_from_video(video_file_path, output_folder_path, target_width=1024, target_height=512):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Open video file
    cap = cv2.VideoCapture(video_file_path)

    # Get total number of frames in video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up progress bar
    progress_bar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

    # Read frames from video file and save them as PNG image files
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize frame to target size
            resized_frame = cv2.resize(frame, (target_width, target_height))

            # Write resized frame as a PNG image file
            output_file_path = os.path.join(output_folder_path, f"frame_{frame_number:06d}.png")
            cv2.imwrite(output_file_path, resized_frame)

            # Increment frame number and update progress bar
            frame_number += 1
            progress_bar.update(1)
        else:
            break

    # Release video file, clean up, and close progress bar
    cap.release()
    cv2.destroyAllWindows()
    progress_bar.close()


if __name__ == "__main__":

    video_file_path = "city.mp4"
    output_folder_path = "demoVideo/custom"
    extract_frames_from_video(video_file_path, output_folder_path)

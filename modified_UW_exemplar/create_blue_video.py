from utils.util import (batch_lab2rgb_transpose_mc, folder2vid, mkdir_if_not,
                        save_frames, tensor_lab2rgb, uncenter_l)
import os

if __name__ == "__main__":
    test_data = "../data_root/test_data"
    for video in os.listdir(test_data):
        frames_path = os.path.join(test_data, video, "blue")
        video_name = "blue_video.avi"
        folder2vid(image_folder=frames_path, output_dir=os.path.join(test_data, video), filename=video_name)

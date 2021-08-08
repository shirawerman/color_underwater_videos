import cv2
import os
from os.path import join
import argparse

# import sys
# sys.path.append(r'W:\colorization\exemplar\Deep-Exemplar-based-Video-Colorization')
# sys.path.append(r'..')

import lib.TestTransforms as transforms
from utils.util_distortion import CenterPad, Normalize, RGB2Lab, ToTensor
import torchvision.transforms as transform_lib


parser = argparse.ArgumentParser()
parser.add_argument("--path",
                    default='./input/video')
parser.add_argument("--output_path",
                    default='./input/frames')
#### The videos need to be in a folder of videos, it opens new folder in the parent directory with the names of the videos

"""

:param path: path to video file
:param output_path: path to output directory
:return:
"""
# vid_path = path
# for file in os.listdir(vid_path):

vidcap = cv2.VideoCapture(path)
# name = os.path.splitext(clip)[0]


if not os.path.exists(output_path):
os.mkdir(output_path, mode)

success,image = vidcap.read()
count = 0

while success:
save_to = os.path.join(output_path, "{:04d}.jpg".format(count))

# transform image
image = transform(image)

if not os.path.exists(save_to):
  print("Save: ", save_to)
  cv2.imwrite(save_to, image)     # save frame as JPEG file

success,image = vidcap.read()
count += 1



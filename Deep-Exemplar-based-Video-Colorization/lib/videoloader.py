import sys

sys.path.insert(0, "..")
import os
import random

import cv2
import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image
from skimage import color
from torch.autograd import Variable
from utils.flowlib import read_flow
from utils.util_distortion import CenterPad

import lib.functional as F

cv2.setNumThreads(0)


class RGB2Lab(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        return color.rgb2lab(inputs)


class Normalize(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        inputs[0:1, :, :] = F.normalize(inputs[0:1, :, :], 50, 1)
        inputs[1:3, :, :] = F.normalize(inputs[1:3, :, :], (0, 0), (1, 1))
        return inputs


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        outputs = F.to_mytensor(inputs)  # permute channel and transform to tensor
        return outputs


class RandomErasing(object):
    def __init__(self, probability=0.6, sl=0.05, sh=0.6):
        self.probability = probability
        self.sl = sl
        self.sh = sh

    def __call__(self, img):
        img = np.array(img)
        if random.uniform(0, 1) > self.probability:
            return Image.fromarray(img)

        area = img.shape[0] * img.shape[1]
        h0 = img.shape[0]
        w0 = img.shape[1]
        channel = img.shape[2]

        h = int(round(random.uniform(self.sl, self.sh) * h0))
        w = int(round(random.uniform(self.sl, self.sh) * w0))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            img[x1 : x1 + h, y1 : y1 + w, :] = np.random.rand(h, w, channel) * 255
            return Image.fromarray(img)

        return Image.fromarray(img)


class CenterCrop(object):
    """
    center crop the numpy array
    """

    def __init__(self, image_size):
        self.h0, self.w0 = image_size

    def __call__(self, input_numpy):
        if input_numpy.ndim == 3:
            h, w, channel = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0, channel))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0, :
            ]
        else:
            h, w = input_numpy.shape
            output_numpy = np.zeros((self.h0, self.w0))
            output_numpy = input_numpy[
                (h - self.h0) // 2 : (h - self.h0) // 2 + self.h0, (w - self.w0) // 2 : (w - self.w0) // 2 + self.w0
            ]
        return output_numpy


def parse_images(data_root):
    image_pairs = []
    subdirs = sorted(os.listdir(data_root))
    # going over all videos
    for subdir in subdirs:
        path_outer = os.path.join(data_root, subdir)  # path to the data directory of one video
        if not os.path.isdir(path_outer):
            continue

        filename = "pairs_list.txt"  # a table-file of the form <frame1> <frame2> <ref_frame>
        parse_file = os.path.join(path_outer, filename)

        under_water_videos = os.path.join(data_root, subdir, "uw_types")  # path to where all the underwater types are
        if os.path.exists(parse_file):
            with open(parse_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.replace("\n", "")
                    (
                        image1_name,
                        image2_name,
                        reference_video_name,
                    ) = line.split()

                    image1_name = image1_name.split(".")[0]
                    image2_name = image2_name.split(".")[0]
                    reference_video_name = reference_video_name.split(".")[0]

                    flow_forward_name = image1_name
                    # flow_backward_name = image1_name + "_backward"  # naming assumption
                    mask_name = image1_name+ "_mask"

                    for vid_dir in os.listdir(under_water_videos):
                        path_inner = os.path.join(under_water_videos, vid_dir)  # path to specific underwater type

                        item = (
                            image1_name + ".png",  # change from '.jpg'  ## DANA
                            image2_name + ".png",  # change from '.jpg'  ## DANA
                            reference_video_name + ".png",  # change from '.jpg'  ## DANA
                            flow_forward_name + ".flo",
                            # flow_backward_name + ".flo",
                            mask_name + ".pgm",
                            path_outer,
                            path_inner,
                        )
                        image_pairs.append(item)

        else:
            raise (RuntimeError(f"Error when parsing {filename} in subfolders of: " + path_outer + "\n"))

    return image_pairs


class VideosDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        epoch,
        image_size,
        image_transform=None,
        use_google_reference=False,
        real_reference_probability=1,
        nonzero_placeholder_probability=0.5,
    ):
        self.data_root = data_root
        self.image_transform = image_transform
        self.CenterPad = CenterPad(image_size)
        self.ToTensor = ToTensor()
        self.CenterCrop = CenterCrop(image_size)

        assert len(self.data_root) > 0, "find no dataroot"
        self.epoch = epoch
        self.image_pairs = parse_images(self.data_root)
        self.real_len = len(self.image_pairs)
        print("##### parsing image pairs in %s: %d pairs #####" % (data_root, self.real_len))
        self.image_pairs *= epoch
        self.use_google_reference = use_google_reference
        self.real_reference_probability = real_reference_probability
        self.nonzero_placeholder_probability = nonzero_placeholder_probability

    def __getitem__(self, index):
        (
            image1_name,
            image2_name,
            reference_video_name,
            flow_forward_name,
            mask_name,
            path_outer,
            path_inner,
        ) = self.image_pairs[index]
        try:
            I1_uw = Image.open(os.path.join(path_inner, image1_name))
            I2_uw = Image.open(os.path.join(path_inner, image2_name))
            gt_dir_name = 'color_down_png'
            I1_gt = Image.open(os.path.join(path_outer, gt_dir_name, image1_name))
            I2_gt = Image.open(os.path.join(path_outer, gt_dir_name, image2_name))
            I_reference_video = Image.open(os.path.join(path_outer, gt_dir_name, reference_video_name))

            flow_forward = read_flow(os.path.join(path_outer, "flow", flow_forward_name))  # numpy
            mask = Image.open(os.path.join(path_outer, "mask", mask_name))

            # binary mask
            mask = np.array(mask)
            mask[mask < 240] = 0
            mask[mask >= 240] = 1

            # transform
            I1_uw = self.image_transform(I1_uw)
            I2_uw = self.image_transform(I2_uw)
            I1_gt = self.image_transform(I1_gt)
            I2_gt = self.image_transform(I2_gt)
            I_reference_video = self.image_transform(self.CenterPad(I_reference_video))
            flow_forward = self.ToTensor(self.CenterCrop(flow_forward))
            mask = self.ToTensor(self.CenterCrop(mask))

            # if np.random.random() < self.real_reference_probability:
            #     I_reference_output = I_reference_video_real
            #     placeholder = torch.zeros_like(I1)
            #     self_ref_flag = torch.zeros_like(I1)
            # else:
            placeholder = I2_gt if np.random.random() < self.nonzero_placeholder_probability else torch.zeros_like(I1_uw)
            self_ref_flag = torch.ones_like(I1_uw)

            outputs = [
                I1_uw,
                I2_uw,
                I1_gt,
                I2_gt,
                I_reference_video,
                flow_forward,
                mask,
                placeholder,
                self_ref_flag,
            ]

        except Exception as e:
            print("problem in, ", path_outer)
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.image_pairs)))
        return outputs

    def __len__(self):
        return len(self.image_pairs)


def batch_lab2rgb_transpose_mc(img_l_mc, img_ab_mc, nrow=8):
    if isinstance(img_l_mc, Variable):
        img_l_mc = img_l_mc.data.cpu()
    if isinstance(img_ab_mc, Variable):
        img_ab_mc = img_ab_mc.data.cpu()

    if img_l_mc.is_cuda:
        img_l_mc = img_l_mc.cpu()
    if img_ab_mc.is_cuda:
        img_ab_mc = img_ab_mc.cpu()

    assert img_l_mc.dim() == 4 and img_ab_mc.dim() == 4, "only for batch input"

    l_norm, ab_norm = 1.0, 1.0
    l_mean, ab_mean = 50.0, 0
    img_l = img_l_mc * l_norm + l_mean
    img_ab = img_ab_mc * ab_norm + ab_mean
    pred_lab = torch.cat((img_l, img_ab), dim=1)
    grid_lab = vutils.make_grid(pred_lab, nrow=nrow).numpy().astype("float64")
    return (np.clip(color.lab2rgb(grid_lab.transpose((1, 2, 0))), 0, 1) * 255).astype("uint8")

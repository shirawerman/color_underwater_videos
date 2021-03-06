# for more info see: https://li-chongyi.github.io/proj_underwater_image_synthesis.html

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.utils import save_image
import os
import torch
from enum import Enum


# different types of UW degragation
class Type(Enum):
    type_I = [0.85, 0.961, 0.982]
    type_IA = [0.84, 0.955, 0.975]
    type_IB = [0.83, 0.95, 0.968]
    type_II = [0.80, 0.925, 0.94]
    type_III = [0.75, 0.885, 0.89]
    type_1 = [0.75, 0.885, 0.875]
    type_3 = [0.71, 0.82, 0.8]
    type_5 = [0.67, 0.73, 0.67]  # results are not very good..
#     type_7 = [0.62, 0.61, 0.5]  # gives black-ish results
#     type_9 = [0.55, 0.46, 0.29]  # gives black results


def generate_video_underwater(images_path, depth_path, output_path="output", randoms=3):
    """ 
    The function creates an underwater degradation for all the images (video frames) in images_path, using their depths 
        from depth_path.
    The names of the corresponding images should be identical.
    For each water type there are randoms variations to be generated; total output is of size:
        number_of_frames X randoms X number of types
    
    params:
    - images_path - path to directory containing the frames
    - depth_path - path to directory containing the depth of the frames, with names as in images_path.
    - randoms - any numbers as you want(a kind of augmentation
    """
    to_tensor = transforms.ToTensor()
    names = [os.path.splitext(file)[0] for file in os.listdir(images_path)]

    for data in Type:
        deep = 5 - 2 * torch.rand(randoms, 1)
        horization = 15 - 14.5 * torch.rand(randoms, 1)
        type = data.value
        for filename in names:
            filename = filename.split('\n')[0]
            im = to_tensor(Image.open(os.path.join(images_path, f'{filename}.png')))
            depth = to_tensor(ImageOps.grayscale(Image.open(os.path.join(depth_path, f'{filename}.png')))).squeeze()
            width, height = depth.size()
            # todo should resize?

            A = torch.zeros(3, 1)
            t = torch.zeros(3, width, height)
            for j in range(randoms):
                vid_path = os.path.join(output_path, f'{data.name}_{deep[j]}_{horization[j]}')
                if not os.path.exists(vid_path):
                    os.makedirs(vid_path)

                A[0, :] = 1.5 * type[0]**deep[j]
                A[1, :] = 1.5 * type[1]**deep[j]
                A[2, :] = 1.5 * type[2]**deep[j]
                t[0, :, :] = type[0]**(depth * horization[j])
                t[1, :, :] = type[1]**(depth * horization[j])
                t[2, :, :] = type[2]**(depth * horization[j])

                output = torch.zeros_like(im)
                output[0, :, :] = A[0] * im[0, :, :] * t[0, :, :] + (1 - t[0, :, :]) * A[0]
                output[1, :, :] = A[1] * im[1, :, :] * t[1, :, :] + (1 - t[1, :, :]) * A[1]
                output[2, :, :] = A[2] * im[2, :, :] * t[2, :, :] + (1 - t[2, :, :]) * A[2]

                under_water_im_path = os.path.join(vid_path, f'{filename}.png')
                save_image(output, under_water_im_path)

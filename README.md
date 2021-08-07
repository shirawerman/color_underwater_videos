


# DL4CV project: Underwater Color Restoration for Videos

https://user-images.githubusercontent.com/71815064/128169439-6d3da019-b3fe-46a7-bd57-ceb8a68694c7.mp4

Our goal is to be able to take an underwater video and make it colorful in a temporal-consistent way, so there is no flickering between frames. 

## Code
- blue_filter - a filter that simulates underwater degradation for images, using depth map. Taken from [here](https://li-chongyi.github.io/proj_underwater_image_synthesis.html)
- colorization_network - the network of temporal-consistant color-enhancement for underwater videos. Original network taken from [here](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)

The depth maps for the blue_filter script were created using [Robust Consistent Video Depth Estimation](https://robust-cvd.github.io/), and the flow maps for the colorization_network were created using [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)

## Usage

## Prerequisites

- Python 3.6+
- Nvidia GPU + CUDA, CuDNN

## Installation

First use the following commands to prepare the environment:

```bash
conda create -n ColorVid python=3.6
source activate ColorVid
pip install -r requirements.txt
```

Then, download the pretrained models from [this link](https://drive.google.com/drive/folders/1OxB0G1blnjIDcFQ2Cnt4RfJbP-Iw-QH-?usp=sharing
unzip the files and place the folders 'checkpoints' and 'data' in the 'modified_DEBVC' directory.

## Data Preparation

In order to colorize your own video, it requires to extract the video frames, and provide a reference image as an example.
To extract the frames, you can run
```bash
python vid_to_frames.py --path <vid_path> --output_path <out_path>
```
Note that the default for input and output path are: './input/your_video' and './input/your_frames' correspondingly.

Next, place your reference images into another folder, _e.g._, './input/your_ref'



To run the network, follow the instructions [here](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization) with the exeption that
you don't need to download the pretrained model.

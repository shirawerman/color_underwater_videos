


# DL4CV project: Underwater Color Restoration for Videos

![Sample Tracking](assets/gif_our_vs_original.gif)


Our goal is to be able to take an underwater video and make it colorful in a temporal-consistent way, so there is no flickering between frames. 

## Code
- blue_filter - a filter that simulates underwater degradation for images, using depth map. Taken from [here](https://li-chongyi.github.io/proj_underwater_image_synthesis.html)
- modified_UW_exemplar - the network for temporal-consistant color-enhancement for underwater videos. Based on the network taken from [here](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)

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

Then, download the pretrained models from [this link](https://drive.google.com/drive/folders/1OxB0G1blnjIDcFQ2Cnt4RfJbP-Iw-QH-?usp=sharing),
unzip the files and place the folders 'checkpoints' and 'data' in the 'modified_DEBVC' directory.

Next, run:

```bash
cd modified_UW_exemplar
```

## Data Preparation

In order to colorize your own video, it requires to extract the video frames, and provide a reference image as an example.
To extract the frames, you can run
```bash
python vid_to_frames.py --path <vid_path> --output_path <out_path>
```
Note that the default for input and output path are: './input/video' and './input/frames' correspondingly.

Next, you need a reference frame.  This can be done offline using existing color enhancement methods, e.g. Dive+.

Place your reference images in a directory named 'ref', _e.g._, './input/ref'

**Note that 'frames' and 'ref' must be in the same directory**

Now, run:

```bash
python test.py --test_dir <test_dir> --output_dir <output_dir>
```
Where test_dir is the directory where both 'frames' and 'ref' are located (default: './input') and output_dir is the name of the output folder (default: 'output'), such that the output video will be in <test_dir>/<output_dir>.

## Comparison with commonly used commercial app

![Sample Tracking](assets/ours_vs_divep.gif)


One can clearly see the frame-to-frame consistency in the left (ours) vs. the flickering in the right (Dive+).

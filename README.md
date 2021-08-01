# DL4CV project: Underwater Color Restoration for Videos

Our goal is to be able to take an underwater video and make it colorful in a temporal-consistent way, so there is no flickering between frames. 

## Networks in use
- blue_filter - a filter that simulates underwater degradation for images, using depth map. Taken from [here](https://li-chongyi.github.io/proj_underwater_image_synthesis.html)
- colorization_network - the network of temporal-consistant color-enhancement for underwater videos. Original network taken from [here](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)
The depth maps for the blue_filter script were created using [Robust Consistent Video Depth Estimation](https://robust-cvd.github.io/), and the flow maps for the colorization_network were created using [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch)



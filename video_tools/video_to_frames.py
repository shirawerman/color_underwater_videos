import cv2
import os


#### The videos need to be in a folder of videos, it opens new folder in the parent directory with the names of the videos


vid_path = './videos/'
for file in os.listdir(vid_path):
    vidcap = cv2.VideoCapture(vid_path + file)
    name = os.path.splitext(file)[0]
    mode = 0o666
    os.mkdir(name, mode)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(name + "/frame%d.jpg" % count, image)     # save frame as JPEG file      
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1


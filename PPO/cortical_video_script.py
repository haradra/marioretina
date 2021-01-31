import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from retinavision.cortical_functions.cortical_map_image import CorticalMapping

# Set up video capture
vidcap = cv2.VideoCapture(os.path.join(os.getcwd(), 'baselines/PPO/output/mario_video_402.mp4'))
success,image = vidcap.read()

count = 0

# Image arrays
images = []
gray_images = []
transformed_images = []

# Set up retina
cortical_transformer = CorticalMapping()
cortical_transformer.setup_retina()
cortical_transformer.setup_cortex()

while success:
    images.append(image)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_images.append(gray)
    # cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
    success,image = vidcap.read()
    # print('Read a new frame: ', success)
    count += 1

for image in gray_images:
    transformed_images.append(cortical_transformer.backproject_transform(image))

# Set up video writer
height, width, layers = images[0].shape
video = cv2.VideoWriter(os.path.join(os.getcwd(), 'baselines/PPO/output/backproject_video.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height), 0)
for image in transformed_images:
    video.write(image)

cv2.destroyAllWindows()
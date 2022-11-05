import os
import sys
import pathlib

module_path = os.path.abspath(os.path.join('.'))

if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

from utils.processImage import unwarp_image

TEST_IMAGES = pathlib.Path.cwd().joinpath("test_images")

fp=TEST_IMAGES.joinpath("sudoku_wiki.jpg")

image = imread(fp)
image = rgb2gray(image[:,:,:3])

unwarped = unwarp_image(image)

fig, ax = plt.subplots(figsize=(1,1))
ax.imshow(unwarped, interpolation=None, cmap='gray')
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
pos = ax.get_position()
pos.x0 = 0
pos.y0 = 0
pos.x1 = 1
pos.y1 = 1
ax.set_position(pos)

savepath = TEST_IMAGES.joinpath(fp.parts[-1].split('.')[0] + "_unwarped.png")

plt.savefig(savepath, dpi=600)
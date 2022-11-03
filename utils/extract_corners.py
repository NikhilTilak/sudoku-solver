import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray

from utils.processImage import unwarp_image

fp = "test_images\sudoku_wiki.jpg"

image = imread(fp)
image = rgb2gray(image[:,:,:3])*255.0

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
plt.savefig(fp.split('\\')[-1].split('.')[0] + "_unwarped.png", dpi=600)
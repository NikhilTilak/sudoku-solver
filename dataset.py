import pathlib
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray


DATA_DIR = pathlib.Path.cwd().joinpath('data')

def generate_dataset():
    data_image = []
    data_label = []

    for root, dirs, files in os.walk(DATA_DIR, topdown=False):
        for name in files:
            fp=pathlib.Path(root).joinpath(name)
            # print(fp)
            label = int(fp.parent.parts[-1])
            if label not in [0,10]:
                img = imread(fp)
                try:
                    img= rgb2gray(img)
                except ValueError:
                    # print(fp)
                    break
                data_image.append(img)
                data_label.append(label)

    data_image = np.asarray(data_image)
    data_label = np.asarray(data_label)
    return data_image, data_label

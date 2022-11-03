import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage import transform
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.patches import Circle

from solver import sudoku

def get_subimage(img, i, j):
    """returns an image of the sudoku box at index i,j
    i=0-9, j=0-9"""
    L = img.shape[0]
    subimage_size = L//9
    crop_margin = int(0.075*subimage_size) #
    simg=img[i*subimage_size:(i+1)*subimage_size,
              j*subimage_size:(j+1)*subimage_size]
    simg = simg[crop_margin:subimage_size-crop_margin, crop_margin:subimage_size-crop_margin] #crop center
    simg = transform.resize(simg,(28,28),anti_aliasing=False) # resize
    simg=simg/255.0 #normalizing
    thresh = threshold_otsu(simg)#threshold to suppress background
    simg[simg<thresh]=0.0
    simg = 1-simg #invert contrast
    if simg.mean()<1e-5:
        simg[:]=0
    return simg

def plot_subimages(im: np.array):
    fig, ax = plt.subplots(9,9)
    for i in range(9):
        for j in range(9):
            simg = get_subimage(im, i,j)
            ax[i,j].imshow(simg, cmap='gray')
            ax[i,j].axes.xaxis.set_ticklabels([])
            ax[i,j].axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    plt.show()
    return True 


coords=[]
def get_corners(im: np.array):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im, cmap='gray')
    
    def extract_coords(event):
            #extract the coordinates of the 4 corners
            # if event.inaxes==ax:
            ix = np.rint(event.xdata)
            iy = np.rint(event.ydata)

            global coords
            coords.append([ix, iy])
            print(coords)
            
            if len(coords)==4:
                fig.canvas.mpl_disconnect(cid)
                plt.close()
                    
    cursor = Cursor(ax, useblit=True, horizOn=True, vertOn=True, color='red', lw=1)
    cid = fig.canvas.mpl_connect('button_press_event', extract_coords)
    plt.show()


def unwarp_image(im: np.array):
    """applies a homomorphic transform to input and returns an unwarped image. 
    Asks user to select the corners of Sudoku in a clockwise order starting from top left corner"""

    get_corners(im)
    print('got corners')

    global coords

    top_left=coords[0]
    top_right=coords[1]
    bottom_right=coords[2]
    bottom_left=coords[3]

    

    FINAL_SIZE = 540 # chose a factor of 9 to make subimages easier to standardize.
    original_points = np.array([top_left, top_right, bottom_left, bottom_right])
    final_points = np.array([[0,0],[FINAL_SIZE,0],[0,FINAL_SIZE],[FINAL_SIZE, FINAL_SIZE]])
    holomorphic_transform= transform.estimate_transform('projective', original_points, final_points)
    unwarped = transform.warp(im, holomorphic_transform.inverse)[:FINAL_SIZE, :FINAL_SIZE]

    plt.imshow(unwarped, cmap='gray')
    plt.show()

    return unwarped


def process_image(fp: str, unwarp=False):
    """Takes filepath to image of sudoku and returns a sudoku object"""

    image = imread(fp)
    image = rgb2gray(image[:,:,:3])*255.0

    if unwarp:
        transformed_image = unwarp_image(image)
    else:
        transformed_image = transform.resize(image,(540,540),anti_aliasing=False)

    sub_images = []
    num_to_find =list(range(81))
    count=0
    for i in range(9):
        for j in range(9):
            simg=get_subimage(transformed_image, i,j)
            sub_images.append(simg)
            if simg.mean()==0:
                num_to_find.remove(count)
            count+=1

    sub_images = np.asarray(sub_images)

    # model = keras.models.load_model(pathlib.Path.cwd().parent.joinpath('models','model.h5'))
    model = keras.models.load_model("my_model")

    sudoku_predictions = model.predict(sub_images)

    # input_array = np.asarray([pred.argmax() if (pred.max()>0.8) else 0 for pred, idx in sudoku_predictions]).reshape(9,9)
    input_array = np.asarray([sudoku_predictions[i].argmax() if ((sudoku_predictions[i].max()>0.8) and (i in num_to_find)) else 0 for i in range(81)]).reshape(9,9)

    return sudoku(input_array)



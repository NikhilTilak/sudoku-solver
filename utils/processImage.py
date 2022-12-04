import os
import pathlib
import numpy as np
import cv2 as cv
from scipy.ndimage import median_filter
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian, threshold_mean
from skimage import transform
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.patches import Circle

from utils.solver import sudoku
from utils.ocr import get_text_from_image



# def get_subimage(img, i, j):
#     """returns an image of the sudoku box at index i,j
#     i=0-9, j=0-9"""
#     L = img.shape[0]
#     subimage_size = L//9
#     crop_margin = int(0.1*subimage_size) # this is to avoid having any lines at the edges of the subimage
#     simg=img[i*subimage_size:(i+1)*subimage_size,
#               j*subimage_size:(j+1)*subimage_size]
#     simg = simg[crop_margin:subimage_size-crop_margin, crop_margin:subimage_size-crop_margin] #crop center
    
   
#     thresh = threshold_otsu(simg)#threshold to suppress background
#     simg[simg<thresh]=0.0 # background=0 (brighter)
#     simg[simg>=thresh]=1.0 # digit=1

#     simg = gaussian(simg, sigma=1, mode='constant', cval=1)
    
#     if simg[subimage_size//4:subimage_size//2,subimage_size//4:subimage_size//2].mean()==1: simg[:]=0 # if center is purely 1, empty
#     if simg[subimage_size//4:subimage_size//2,subimage_size//4:subimage_size//2].mean()==0: simg[:]=0 # if center is purely 0, empty

#     simg = transform.resize(simg,(28,28),anti_aliasing=True) # resize 

#     simg = 1-simg #invert contrast such that background=1 (darker)
    
#     return simg

def get_subimage(im, i, j):
    """returns an image of the sudoku box at index i,j
    i=0-9, j=0-9"""
    img = im.copy()
    L = img.shape[0]
    subimage_size = L//9

    simg=img[i*subimage_size:(i+1)*subimage_size,
                    j*subimage_size:(j+1)*subimage_size]

    smaller_crop = int(0.25*subimage_size) # this is to avoid having any lines at the edges of the subimage
    center_crop = simg[smaller_crop:subimage_size-smaller_crop, smaller_crop:subimage_size-smaller_crop] #crop center

    larger_crop = int(0.1*subimage_size) # this is to avoid having any lines at the edges of the subimage
    simg= simg[ larger_crop:subimage_size- larger_crop,  larger_crop:subimage_size- larger_crop] #crop center
   
    if np.std(center_crop) <50.0:
            simg = np.ones(simg.shape)*255.0
    else:
        thresh = threshold_otsu(simg)#threshold to suppress background
        simg[simg<thresh]=0.0 # background=0 (brighter)
        simg[simg>=thresh]=255.0 # digit=1

    simg = 255.0-simg #invert contrast such that background=1 (darker)
    
    return simg

def plot_subimages(im: np.array):
    """Plots the subimages as a 9x9 grid"""
    fig, ax = plt.subplots(9,9, figsize=(5,5))
    for i in range(9):
        for j in range(9):
            simg = get_subimage(im, i,j)
            ax[i,j].imshow(simg, cmap='viridis')
            ax[i,j].text(0.25,0.25,f'{(simg[:]==0).all()}')
            ax[i,j].set_aspect('equal')
            ax[i,j].set_xticks([])
            ax[i,j].set_yticks([])
            ax[i,j].axes.xaxis.set_ticklabels([])
            ax[i,j].axes.yaxis.set_ticklabels([])
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.show()

def get_all_subimages(im):
    rows = []
    for i in range(9):
        cols = []
        for j in range(9):
            simg = get_subimage(im, i,j)
            cols.append(simg)
            if j<8: cols.append(np.ones((simg.shape[0], 5))*255.0)
        rows.append(np.concatenate(cols, axis=1))
        if i<8: rows.append(np.ones((5, rows[0].shape[1]))*255.0)
    return np.concatenate(rows, axis=0)

def stitch_subimages(im: np.array):
    """ takes a 540x540 transformed image and outputs stitched image of numbers to find."""
    sub_images = []
    num_to_find =list(range(81))
    count=0
    for i in range(9):
        for j in range(9):
            simg=get_subimage(im, i,j)
            if (simg[:]==0).all(): # empty image if only 0's.
                num_to_find.remove(count)
            else: # non-empty image
                simg = transform.resize(simg,(128,128),anti_aliasing=True) # resize 
                sub_images.append(simg[:,10:118])
                # sub_images.append(gaussian(simg, sigma=0, mode='constant', cval=1))
            count+=1
    stitched = np.concatenate(sub_images, axis=1)
    return stitched, num_to_find


coords=[]
def get_corners(im: np.array):

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(im, cmap='gray')
    
    def extract_coords(event):
            #extract the coordinates of the 4 corners
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

    

    FINAL_SIZE = 900 # chose a factor of 9 to make subimages easier to standardize.
    original_points = np.array([top_left, top_right, bottom_left, bottom_right])
    final_points = np.array([[0,0],[FINAL_SIZE,0],[0,FINAL_SIZE],[FINAL_SIZE, FINAL_SIZE]])
    holomorphic_transform= transform.estimate_transform('projective', original_points, final_points)
    unwarped = transform.warp(im, holomorphic_transform.inverse)[:FINAL_SIZE, :FINAL_SIZE]

    plt.imshow(unwarped, cmap='gray')
    plt.show()

    return unwarped


def process_image(fp, unwarp=False, use_ocr=True, TEST_IMAGES=pathlib.Path("D:\Data Science\sudoku-solver\\test_images")):
    """Takes filepath to image or numpy array of sudoku and returns a sudoku object"""
    # TEST_IMAGES = pathlib.Path.cwd().parent.joinpath("test_images")
    if type(fp)==np.ndarray:
        image=fp
    else:
        image = imread(fp)
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.fastNlMeansDenoising(np.uint8(image))

    if unwarp:
        transformed_image = transform.resize(unwarp_image(image),(540,540),anti_aliasing=True)
    else:
        transformed_image = image

    if use_ocr:
        #using OCR API for digit identification
        stitched, num_to_find = stitch_subimages(transformed_image)
        print(num_to_find)
        plt.imshow(1-stitched, cmap='gray')
        plt.axis('off')
        plt.savefig(TEST_IMAGES.joinpath("tempfile.jpg"))
        nums_in_image = get_text_from_image(TEST_IMAGES.joinpath("tempfile.jpg"))
        nums_in_image= [int(n) for n in list(nums_in_image)]

        if len(nums_in_image)<len(num_to_find):
            print("error in identifying empty cell")
        if len(nums_in_image)>len(num_to_find):
            print("error in OCR")
            

        input_array=np.zeros(81)
        for i,v in enumerate(num_to_find):
                input_array[v] = nums_in_image[i]
        input_array = input_array.reshape(9,9)
        os.remove(TEST_IMAGES.joinpath("tempfile.jpg"))

    else:
        # using the neural network model for digit predictions
        sub_images = []
        num_to_find =list(range(81))
        count=0
        for i in range(9):
            for j in range(9):
                simg=get_subimage(transformed_image, i,j)
                sub_images.append(simg)
                if simg.mean()==1: # Empty image
                    num_to_find.remove(count)
                count+=1
        # use keras model to predict digits
        model = keras.models.load_model("..\my_model")
        sub_images = np.asarray(sub_images)
        sudoku_predictions = model.predict(sub_images)
        
        input_array = np.asarray([sudoku_predictions[i].argmax() if ((sudoku_predictions[i].max()>0.8) and (i in num_to_find)) else 0 for i in range(81)]).reshape(9,9)
        

    return sudoku(input_array)





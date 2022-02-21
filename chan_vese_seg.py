import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
from skimage.segmentation import mark_boundaries
from PIL import Image, ImageFilter
import os

'''
myImage = Image.open(r"./Temp_Images/RGB/08162016_Mississippi_River_at_Baton_Rouge_LA.tif_RGB.png")
myImage2 = Image.open(r"./Temp_Images/GT/08162016_Mississippi_River_at_Baton_Rouge_LA.tif_RGB.png")

myImage = myImage.resize((256, 256), Image.ANTIALIAS)
myImage2 = myImage2.resize((256, 256), Image.ANTIALIAS)

myImage.show()
myImage2.show()
'''

def get_data():

    gt = []
    rgb = []
    for e in sorted(os.listdir('D:\PythonLearning\ChanVese\Temp_Images_2/RGB')):
        rgb.append(Image.open('D:\PythonLearning\ChanVese\Temp_Images_2/RGB/' + e).resize((256, 256), Image.NEAREST).convert('RGB'))
        print(e)

    print('------------------------------------------')
    for e in sorted(os.listdir('D:\PythonLearning\ChanVese\Temp_Images_2/GT')):
        gt.append(Image.open('D:\PythonLearning\ChanVese\Temp_Images_2/GT/' + e).resize((256, 256), Image.NEAREST).convert('RGB'))
        print(e)

    return gt, rgb


input1, input2 = get_data()


def chan_vese_seg(images, images2):

    float_images = []
    segments = []
    for e in images:

        temp = np.array(e)
        float_images.append(temp)
        segments.append(chan_vese(rgb2gray(temp), mu=0.04, lambda1=1, lambda2=2, tol=1e-8,
                   dt=0.5, init_level_set="checkerboard", extended_output=True))

    float_images_gt = []
    for e in images2:

        temp = np.array(e)
        float_images_gt.append(temp)

    #fig, axes = plt.subplots(3, 8, figsize=(8, 8))
    #ax = axes.flatten()

    fig, ax = plt.subplots(3, 8, figsize=(20, 20), sharex=True, sharey=True)

    #Image1
    ax[0, 0].imshow(float_images[0], cmap="gray")
    ax[0, 0].set_title("RGB Image 1", fontsize=12)
    ax[1, 0].imshow(float_images_gt[0], cmap="gray")
    ax[1, 0].set_title("GT Image 1", fontsize=12)
    ax[2, 0].imshow(segments[0][0], cmap="gray")
    title = "{} Iterations".format(len(segments[0][2]))
    ax[2, 0].set_title(title, fontsize=12)


    #Image2
    ax[0, 1].imshow(float_images[1], cmap="gray")
    ax[0, 1].set_title("RGB Image 2", fontsize=12)
    ax[1, 1].imshow(float_images_gt[1], cmap="gray")
    ax[1, 1].set_title("GT Image 2", fontsize=12)
    ax[2, 1].imshow(segments[1][0], cmap="gray")
    title = "{} Iterations".format(len(segments[1][2]))
    ax[2, 1].set_title(title, fontsize=12)

    #Image3
    ax[0, 2].imshow(float_images[2], cmap="gray")
    ax[0, 2].set_title("RGB Image 3", fontsize=12)
    ax[1, 2].imshow(float_images_gt[2], cmap="gray")
    ax[1, 2].set_title("GT Image 3", fontsize=12)
    ax[2, 2].imshow(segments[2][0], cmap="gray")
    title = "{} Iterations".format(len(segments[2][2]))
    ax[2, 2].set_title(title, fontsize=12)

    #Image4
    ax[0, 3].imshow(float_images[3], cmap="gray")
    ax[0, 3].set_title("RGB Image 4", fontsize=12)
    ax[1, 3].imshow(float_images_gt[3], cmap="gray")
    ax[1, 3].set_title("GT Image 4", fontsize=12)
    ax[2, 3].imshow(segments[3][0], cmap="gray")
    title = "{} Iterations".format(len(segments[3][2]))
    ax[2, 3].set_title(title, fontsize=12)

    #Image5
    ax[0, 4].imshow(float_images[4], cmap="gray")
    ax[0, 4].set_title("RGB Image 5", fontsize=12)
    ax[1, 4].imshow(float_images_gt[4], cmap="gray")
    ax[1, 4].set_title("GT Image 5", fontsize=12)
    ax[2, 4].imshow(segments[4][0], cmap="gray")
    title = "{} Iterations".format(len(segments[4][2]))
    ax[2, 4].set_title(title, fontsize=12)

    #Image6
    ax[0, 5].imshow(float_images[5], cmap="gray")
    ax[0, 5].set_title("RGB Image 6", fontsize=12)
    ax[1, 5].imshow(float_images_gt[5], cmap="gray")
    ax[1, 5].set_title("GT Image 6", fontsize=12)
    ax[2, 5].imshow(segments[5][0], cmap="gray")
    title = "{} Iterations".format(len(segments[5][2]))
    ax[2, 5].set_title(title, fontsize=12)

    #Image7
    ax[0, 6].imshow(float_images[6], cmap="gray")
    ax[0, 6].set_title("RGB Image 7", fontsize=12)
    ax[1, 6].imshow(float_images_gt[6], cmap="gray")
    ax[1, 6].set_title("GT Image 7", fontsize=12)
    ax[2, 6].imshow(segments[6][0], cmap="gray")
    title = "{} Iterations".format(len(segments[6][2]))
    ax[2, 6].set_title(title, fontsize=12)

    #Image8
    ax[0, 7].imshow(float_images[7], cmap="gray")
    ax[0, 7].set_title("RGB Image 8", fontsize=12)
    ax[1, 7].imshow(float_images_gt[7], cmap="gray")
    ax[1, 7].set_title("GT Image 8", fontsize=12)
    ax[2, 7].imshow(segments[7][0], cmap="gray")
    title = "{} Iterations".format(len(segments[7][2]))
    ax[2, 7].set_title(title, fontsize=12)

    plt.savefig("Chan_Vese_Results.png")

    fig.tight_layout()
    plt.show()


chan_vese_seg(input2, input1)
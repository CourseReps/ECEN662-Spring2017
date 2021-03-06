# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:52:11 2017

@author: fatadama
@brief: use the INI file to load test images into numpy arrays

HOW TO USE
----
- Download the training data folders into the same folder
- Modify "settings.ini" to indicate the relative file path from this script's
    folder to the root directory of the training data
- Run this file to perform test reads from the synthetic and scene image sets

Example use: download the training sets to "~/ecen_data/TrainingSetScenes" and 
    "~/ecen_data/TrainingSetSynthetic". Download this file to "~/ecen_python". 
    Modify the "filepath" value in settings.ini to "../ecen_data". Then run
    this script to test/.
"""

import numpy as np
from scipy import ndimage
from scipy import misc
import sys
import os

# python 2 or 3 check
if sys.version_info > (3,0):
    import configparser
else:
    import ConfigParser as configparser

inifile = 'settings.ini'

# get path to image file
config = configparser.ConfigParser()
config.read(inifile)
# GLOBAL filepath
FilePath = config.get('directory','filepath')
# GLOBAL list of natural scenes - only load once
SceneList = os.listdir(FilePath+'/TrainingSetScenes/')

print("To use this file, the relative filepath to the root image directory must be: %s" % (FilePath))

def loadImg(filepath,iname,sz=None):
    """Load an image from the data set
    
    Parameters
    ----
    filepath : string
        filepath with trailing backslash
    iname : string
        image name with no leading backslash
    """
    # load image
    img = ndimage.imread(filepath+iname)
    # resize image if desired
    if sz is not None:
        if sz[0] < sz[1]:
            if img.shape[0] < img.shape[1]:
                im = misc.imresize(img,size=sz)
            else:
                im = misc.imresize(img.transpose(),size=sz)
        else:
            if img.shape[0] > img.shape[1]:
                im = misc.imresize(img,size=sz)
            else:
                im = misc.imresize(img.transpose(),size=sz)
        return im
    return img
    
def loadScene(i,sz=None):
    """Load an image from the set of natural images
    
    These are not ordered sequentially, so we use the system ordering
    
    Parameters
    ----
    i : int or string
        If string, load an image with this name
        If int, load the ith image as ordered by the system operator. i is 1-indexed.
    """
    if type(i) == str:
        iname = i
    else:
        iname = SceneList[i-1]
    img = loadImg(FilePath+'/TrainingSetScenes/',iname,sz)
    return img

def loadSynthetic(i,sz=None,spec=False):
    """ Load an image from the training set
    
    Parameters
    ----
    i : int or string
        If i is an integer, attempt to load the ith training image (index from 1)
        If i is a string, attempt to load the training image with filename 'i'
        E.g., loadTraining(20) or loadTraining("image20")
    spec : boolean
        Pass in True to load the synthetic image with specular reflections
    """
    if type(i) == int:
        if spec == False:
            iname = 'image%d.png' % (i)
        else:
            iname = 'image%d_spec.png' % (i)
    else:
        iname = i
    img = loadImg(FilePath+'/TrainingSetSynthetic/',iname,sz)
    return img
    
def im2intensity(im):
    """ Convert an RGB image to intensity space
    
    Parameters
    ----
    im : N x M x 3 array
        input image in RGB space
    
    Returns
    ----
    io : N x M array
        output in intensity space
    """
    io = 0.2989 * im[:,:,0] + 0.5870 * im[:,:,1] + 0.1140 * im[:,:,2]
    #io = np.sqrt(np.sum(np.power(im.astype(float),2.0),axis=2)/3.0)
    '''
    io = np.zeros((im.shape[0],im.shape[1]))
    for k in range(im.shape[0]):
        for j in range(im.shape[1]):
            io[k,j] = np.sqrt(np.sum(np.power(im[k,j,:],2.0)))
    '''
    return io

def test():
    """
    Load synthetic image 1 and natural scene 2, and plot them using matplotlib
    """
    print("Load synthetic image")
    img = loadSynthetic(1)
    import matplotlib.pyplot as plt
    plt.figure("Synthetic image 1")
    plt.imshow(img)
    plt.show()
    print("Load natural image")
    img = loadScene(2)
    import matplotlib.pyplot as plt
    plt.figure("Natural scene 2")
    plt.imshow(img)
    plt.show()
    return
    
if __name__ == '__main__':
    test()
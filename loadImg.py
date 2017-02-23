# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 15:52:11 2017

@author: fatadama
@brief: use the INI file to load test images into numpy arrays
"""

import numpy as np
from scipy import ndimage
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

def loadImg(filepath,iname):
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
    return img
    
def loadNatural(i):
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
    img = loadImg(FilePath+'/TrainingSetScenes/',iname)
    return img

def loadSynthetic(i,spec=False):
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
    img = loadImg(FilePath+'/TrainingSetSynthetic/',iname)
    return img

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
    img = loadNatural(2)
    import matplotlib.pyplot as plt
    plt.figure("Natural Scene 2")
    plt.imshow(img)
    plt.show()
    return
    
if __name__ == '__main__':
    test()
import os
import glob


import cv2
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import cauchy, expon, norm, laplace, poisson, gamma, lognorm

scene_drive = "/src/images/TrainingSetScenes/"
synthetic_drive = "/src/images/TrainingSetSynthetic/"

home = os.environ['HOME']

scene_drive = home + scene_drive
synthetic_drive = home + synthetic_drive

scene_image_list = glob.glob(scene_drive+"*.jpg")
synethic_image_list = glob.glob(synthetic_drive+"*.png")

plt.close("all")


syn_img = cv2.imread(synethic_image_list[np.random.randint(0,len(synethic_image_list))],0)
scene_img = cv2.imread(scene_image_list[np.random.randint(0,len(scene_image_list))],0)

imgs = (syn_img,scene_img)

dist_funs = (expon,cauchy)
dist_params = (dict(loc=0,scale=60),dict(loc=0,scale=100) )

for img, dist_fun, dist_params in zip(imgs, dist_funs, dist_params):
    
    fig = plt.figure()

    width = 1000
    if img.shape[0] > width:
        new_height =  int(width*1./img.shape[1]*img.shape[0])
        img = cv2.resize(img, (width, new_height)) 
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    #sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    plt.subplot(2,2,1)
    plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    #plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(1,2,2),plt.imshow(sobely,cmap = 'gray')
    #plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()
    
    
#    f = np.fft.fft2(img)
#    fshift = np.fft.fftshift(f)
#    magnitude_spectrum = 20*np.log(np.abs(f))
    plt.subplot(2,2,3)
    lims = (-1000, 1000)
    n, bins, patches = plt.hist(sobelx.flatten(), 100, normed=1, facecolor='green', alpha=0.75, range=lims)
    x = np.linspace(bins[0], bins[-1], 1000)
    

    pdf_vals = dist_fun.pdf(x,**dist_params)
    plt.plot(x,pdf_vals,color="r")
    plt.xlim(*lims)
#    plt.subplot(2,2,4)
#    n, bins, patches = plt.hist(magnitude_spectrum.flatten(), 100, normed=1, facecolor='green', alpha=0.75)
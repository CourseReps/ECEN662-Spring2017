# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: fatadama
@brief: load images and use numpy to compute the image gradients
"""

import numpy as np
from loadImg import *
import matplotlib.pyplot as plt

# GLOBAL: training set size
TrainingSize = 23

def loadImgPlotGradient():
    """ Load a scene and a synthetic image and plot the intensity gradients for
    each
    """
    im = loadScene(1)
    io = im2intensity(im)
    gra = np.gradient(io)[0]
    flatGra = gra.flatten()
    
    im2 = loadSynthetic(1)
    io2 = im2intensity(im2)
    gra2 = np.gradient(io2)[0]
    flatSyn = gra2.flatten()
    
    fig,ax = plt.subplots(1,2)
    
    ax[0].imshow(gra)#.set_cmap('gray')
    ax[1].imshow(gra2)#.set_cmap('gray')
    
    print(np.mean(flatGra))
    print(np.mean(flatSyn))
    
    #ax[0].set_title('Histogram for gradient of scene 1')
    #ax[0].hist(flatGra)
    
    #ax[1].set_title('Histogram for gradient of synthetic 1')
    #ax[1].hist(flatSyn)
    
    plt.show()

def testMeanGradient():
    """ Use the mean intensity gradient as a discriminator
    """    
    dat1 = np.zeros((TrainingSize,2))
    
    for k in range(TrainingSize):
        im2 = loadScene(k+1)
        io2 = im2intensity(im2)
        gra2 = np.gradient(io2)[0]
        print(np.mean(gra2.flatten()))
        dat1[k,1] = np.mean(gra2.flatten())
    print('-------')
    for k in range(TrainingSize):
        im = loadSynthetic(k+1)
        io = im2intensity(im)
        gra = np.gradient(io)[0]
        print(np.mean(gra.flatten()))
        dat1[k,0] = np.mean(gra.flatten())
        
    # mean gradient as discriminator
    bins = np.arange(-0.5,0.50001,0.025)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    
    synthetichist = np.array(np.histogram(dat1[:,0],bins=bins)[0]).astype(float)/TrainingSize
    scenehist = np.array(np.histogram(dat1[:,1],bins=bins)[0]).astype(float)/TrainingSize
    likelihood = np.array(synthetichist).astype(float)/np.array(scenehist).astype(float)
    likelihood[np.where(scenehist==0)[0]] = 0.0
        
    #plt.figure()
    fig,ax = plt.subplots(1,3)
    ax[0].hist(dat1[:,1],bins=bins,rwidth=1.0,label='scene')
    ax[0].hist(dat1[:,0],bins=bins,rwidth=0.5,label='synthetic')
    ax[0].legend()
    
    ax[1].plot(barx,likelihood,'k--x')

    # find the discriminator
    syntheticx = np.where(likelihood >= 1.0)[0]
    scenex = np.setdiff1d(range(len(likelihood)),syntheticx)
    
    # probability of error using this discriminator, assuming the histogram is CDF
    # assume equal priors
    perr = np.zeros(scenehist.shape)
    perr[syntheticx] = scenehist[syntheticx]*0.5
    perr[scenex] = synthetichist[scenex]*0.5
    ax[2].bar(barx,perr,width=binwidth)
    ax[2].set_title('Predicted probability of error with equal priors')
    print("Total predicted probability of error: %f" % (np.sum(perr*binwidth)))
    # print synethetic image window    
    print(barx[syntheticx])
    print("Synthetic classifier: (%f,%f)" % (barx[syntheticx[0]],barx[syntheticx[-1]]))
    
    plt.show()
    
def trialMeanGradient(xlower=-0.0625,xupper=0.0625):
    falsePositives = 0
    for k in range(len(SceneList)):
        im = loadScene(k+1)
        io = im2intensity(im)
        gra = np.gradient(io)[0]
        metric = np.mean(gra.flatten())
        print(k,metric)
        if (metric >= xlower) and (metric <= xupper):
            # synthetic image detected
            falsePositives = falsePositives + 1
            print("False positive")
    print("%d false positives out of %d" % (falsePositives,len(SceneList)))
    trueDetection = 0
    for k in range(99):
        im = loadSynthetic(k+1)
        io = im2intensity(im)
        gra = np.gradient(io)[0]
        metric = np.mean(gra.flatten())
        print(k,metric)
        if (metric >= xlower) and (metric <= xupper):
            # synthetic image detected
            trueDetection = trueDetection + 1
            print("True detection")
    print("%d true detections out of %d" % (trueDetection,99))
    pass
    
    
def testPlotFFT(iu=1):
    """ Plot the FFT and a histogram of FFT values for scene and synthetic 
    image at index 'iu'
    """
    im2 = loadSynthetic(iu)
    im = loadScene(iu)
    ff2 = np.fft.fft2(im2)
    ff = np.fft.fft2(im)
    logff = np.log10(np.abs(ff))
    logff2 = np.log10(np.abs(ff2))
    
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(logff)
    ax[0].set_title('Scene')
    ax[2].hist(logff.ravel(),label='Scene')
    
    ax[1].imshow(logff2)
    ax[1].set_title('Synthetic')
    ax[2].hist(logff2.ravel(),label='Synthetic')
    ax[2].legend()
    plt.show()
    return
    
def testFFT():
    """ Evaluate using histogrammed FFT for scene and synthetic image
    discrimination
    """
    
    bins = np.arange(-10,10.00001,0.5)
    #bins = [-10,-6,-2,2,6,10]
    
    barx = 0.5*np.diff(bins)+bins[:-1]
    binwidth = np.mean(np.diff(bins))
        
    hist1 = np.zeros((len(bins)-1,))
    hist2 = np.zeros((len(bins)-1,))
    np.set_printoptions(precision=2,suppress=True)
    for k in range(TrainingSize):
        im = loadScene(k+1)
        ff = np.fft.fft2(im)
        logff = np.log10(ff)
        hist = np.histogram(logff,bins=bins)[0]
        hist = hist.astype(float)/(logff.shape[0]*logff.shape[1])
        print(hist)
        hist1 = hist1 + hist
    print('-----')
    for k in range(TrainingSize):
        im = loadSynthetic(k+1)
        ff = np.fft.fft2(im)
        logff = np.log10(ff)
        hist = np.histogram(logff,bins=bins)[0]
        hist = hist.astype(float)/(logff.shape[0]*logff.shape[1])
        print(hist)
        hist2 = hist2+hist
    
    hist1 = hist1.astype(float)/np.sum(hist1)
    hist2 = hist2.astype(float)/np.sum(hist2)
    
    fig,ax = plt.subplots(1,2)
    scene = ax[0].bar(barx,hist1,binwidth,color='blue')
    synthetic = ax[0].bar(barx+0.25*binwidth,hist2,0.5*binwidth,color='red')
    ax[0].legend((scene[0],synthetic[0]),('scene','synthetic'))

    # find the discriminator
    syntheticx = np.where(hist2 >= hist1)[0]
    scenex = np.setdiff1d(range(len(hist1)),syntheticx)
    perr = np.zeros(hist1.shape)
    perr[syntheticx] = hist1[syntheticx]*0.5
    perr[scenex] = hist2[scenex]*0.5
    # plot the probability of error
    ax[1].bar(barx,perr,width=binwidth)
    ax[1].set_title('Predicted probability of error with equal priors')
    print("Total predicted probability of error: %f" % (np.sum(perr*binwidth)))
    
    plt.show()

if __name__ == '__main__':
    #testMeanGradient()
    #testPlotFFT(10)
    #testFFT() 
    trialMeanGradient()
    pass
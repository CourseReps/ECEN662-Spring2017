# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:12:07 2017

@author: fatadama
@brief: Main code for edge detection test
"""
import numpy as np
from loadImg import *
import matplotlib.pyplot as plt
from time import sleep
from scipy import stats

# GLOBAL: training set size
TrainingSize = 23
# FIG WIDTH (total width)
figwidth = 11.0
# FIG HEIGHT (per subplot)
figheight = 2.8
# fontsize
fontsize = 12

scale_scene = 52984.7
loc_scene = 131700
scale_synthetic = 5377.61
loc_synthetic = 35760.8

def tryEdgeDetection():
    """ Load the training set of scenes and synthetic images and plot
    histograms of the gradients for each case
    
    Used to determine a threshold for edge detection
    """
    bins = np.arange(0.0,56.0001,2)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    
    sz = [540,960]
    
    fig,ax = plt.subplots(1,1,figsize=(figwidth,figheight))
    
    valSynthetic = np.zeros((TrainingSize,sz[0],sz[1]))
    valScene = np.zeros((TrainingSize,sz[0],sz[1]))
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        valSynthetic[h,:,:] = np.gradient(im)[0]
    for h in range(TrainingSize):        
        im = im2intensity(loadScene(h+1,sz=sz))
        valScene[h,:,:] = np.gradient(im)[0]
    
    barSynthetic = np.histogram(valSynthetic,bins,normed=True)[0]
    barScene = np.histogram(valScene,bins,normed=True)[0]
    
    scene = ax.bar(bins[:-1],barScene,width=binwidth,color='blue')
    synthetic = ax.bar(bins[:-1]+0.25*binwidth,barSynthetic,width=0.5*binwidth,color='green')
    ax.legend((scene[0],synthetic[0]),('scene','synthetic'))
    ax.set_title('Normalized histogram of pixel gradients')
    ax.set_xlabel('Pixel gradient')
    ax.set_ylabel('PMF')
    plt.tight_layout()
        
    plt.show()
    
def edgeLength():
    """
    Using a hard-coded threshold value, do edge detection. 
    Create normalized histograms of the number of edges for the training set.
    """
    # threshold
    thres = 4.0
    # size to rescale all images to
    sz = [540,960]
    
    valSynthetic = np.zeros((TrainingSize,))
    valScene = np.zeros((TrainingSize,))
    # loop over synthetic
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        gra = np.gradient(im)[0]
        valSynthetic[h] = len(np.where(np.abs(gra) > thres)[0])
    # loop over scenes
    for h in range(TrainingSize):        
        im = im2intensity(loadScene(h+1,sz=sz))
        gra = np.gradient(im)[0]
        valScene[h] = len(np.where(np.abs(gra) > thres)[0])
    
    # fit distributions to data
    scenefit = stats.cauchy.fit(valScene)
    syntheticfit = stats.cauchy.fit(valSynthetic)
    print("Cauchy distribution for scenes: loc=%12.6f, scale=%12.6f" % (scenefit[0],scenefit[1]))
    print("Cauchy distribution for synthetic: loc=%12.6f, scale=%12.6f" % (syntheticfit[0],syntheticfit[1]))
    
    fig,ax = plt.subplots(2,1,figsize=(figwidth,figheight*2))
    
    bins = np.linspace(0.0,400000,10)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    xpdf = np.linspace(bins[0],bins[-1],100)
    barScene = np.histogram(valScene,bins,normed=True)[0]
    
    ax[0].bar(bins[:-1],barScene,width=binwidth,color='blue')
    #ax[0].hist(valScene,rwidth=1,normed=True)
    ax[0].plot(xpdf,stats.cauchy.pdf(xpdf,loc=scenefit[0],scale=scenefit[1]),'k-',linewidth=2)
    ax[0].set_xlim((bins[0],bins[-1]))
    ax[0].set_ylabel('Scene')
    ax[0].set_title('Normalized edge length histograms')
    #ax[0].tick_params(labelsize=fontsize)
    plt.tight_layout()
    
    bins = np.linspace(0,65000,10)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))   
    xpdf = np.linspace(bins[0],bins[-1],100)
    barSynthetic = np.histogram(valSynthetic,bins,normed=True)[0]
    ax[1].bar(bins[:-1],barSynthetic,width=binwidth,color='green')
    ax[1].plot(xpdf,stats.cauchy.pdf(xpdf,loc=syntheticfit[0],scale=syntheticfit[1]),'k-',linewidth=2)
    ax[1].set_xlim((bins[0],bins[-1]))
    ax[1].set_xlabel('Number of edges')
    ax[1].set_ylabel('Synthetic')
    plt.tight_layout()
    #ax[1].hist(valSynthetic,rwidth=1,normed=True)
        
    plt.show()
    
def edgeTrialIm(im,threshold=1.0):
    """ Use the edge detection criterion to classify an image as synthetic (0)
    or natural (1)
    
    Parameters
    ----
    im : 2D array
        a greyscale image, must already be resized to desired size
    threshold : float
        the likelihood ratio == pi0/pi1
    Returns
    ----
    out : bool
        0 == synthetic, 1 == scene
    likelihood : float
        the likelihood ratio
    z : float
        the number of edge pixels
    """
    thres = 4.0
    gra = np.gradient(im)[0]
    count = len(np.where(np.abs(gra) > thres)[0])
    p0 = stats.cauchy.pdf(count,scale=scale_synthetic,loc=loc_synthetic)
    p1 = stats.cauchy.pdf(count,scale=scale_scene,loc=loc_scene)
    likelihood = float(p1)/float(p0)
    if likelihood >= threshold:
        return (1,likelihood,count)
    else:
        return (0,likelihood,count)
    
def trialEdgeLength():
    """
    Loop over all values not in the training set. Classify as synthetic or
    scene using edge detection test and report number of false positives and
    negatives.
    
    Uses equal priors.
    """
    # null hypothesis: normal with scale=38192,loc = 8817
    # alternate hypothesis: cauchy with 153595, loc = 45142

    thres = 4.0    
        
    sz = [540,960]
        
    testingSize0 = 99-TrainingSize
    testingSize1 = len(SceneList)-TrainingSize
    
    valSynthetic = np.zeros((testingSize0,))
    valScene = np.zeros((testingSize1,))
    for h in range(testingSize0):
        im = im2intensity(loadSynthetic(h+TrainingSize,sz=sz))
        gra = np.gradient(im)[0]
        valSynthetic[h] = len(np.where(np.abs(gra) > thres)[0])
    for h in range(testingSize1):
        im = im2intensity(loadScene(h+TrainingSize,sz=sz))
        gra = np.gradient(im)[0]
        valScene[h] = len(np.where(np.abs(gra) > thres)[0])

    # null hypothesis: synthetic    
    p00 = stats.cauchy.pdf(valSynthetic,scale=scale_synthetic,loc=loc_synthetic)
    p01 = stats.cauchy.pdf(valScene,scale=scale_synthetic,loc=loc_synthetic)
    # test hypothesis: scene
    p10 = stats.cauchy.pdf(valSynthetic,scale=scale_scene,loc=loc_scene)
    p11 = stats.cauchy.pdf(valScene,scale=scale_scene,loc=loc_scene)
    
    l0 = p10/p00
    l1 = p11/p01
    
    w0 = np.where(l0 > 1)[0]
    w1 = np.where(l1 < 1)[0]
    
    print("%d false postives out of %d" % (len(w0),len(l0)))
    print("%d false negatives out of %d" % (len(w1),len(l1)))
    """
    fig,ax = plt.subplots(2,1)
    ax[0].plot(valSynthetic,p00,'gx')
    ax[0].plot(valSynthetic,p10,'bd')
    
    ax[1].plot(valScene,p01,'gx')
    ax[1].plot(valScene,p11,'bd')
    
    plt.show()
    """
    return

def edgeTrialCompare():
    """ Compare the likelihood ratios of the two data sets for different values
    of the likelihood threshold
    """
    
    # equal priors
    thres2 = 1.0
    logthres2 = np.log(thres2)
    # unequal priors: prob(synthetic)/prob(scene)
    thres1 = 99.0/float(len(SceneList))
    
    sz = (540,960)
    
    sceneLikelihoods = np.zeros((len(SceneList),))
    xScene = np.zeros((len(SceneList),))
    vScene = np.zeros((len(SceneList),))
    for h in range(len(SceneList)):
        im = im2intensity(loadScene(h+1,sz=sz))
        (vScene[h],sceneLikelihoods[h],xScene[h]) = edgeTrialIm(im,thres2)
    syntheticLikelihoods = np.zeros((99,))
    xSynthetic = np.zeros((99,))
    vSynthetic = np.zeros((99,))
    for h in range(99):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        (vSynthetic[h],syntheticLikelihoods[h],xSynthetic[h]) = edgeTrialIm(im,thres2)
    print("%d false postives out of %d" % (np.sum(vSynthetic > 0),len(vSynthetic)))
    print("%d false negatives out of %d" % (np.sum(vScene < 1),len(vScene)))
    
    fig,ax = plt.subplots(2,1,figsize=(figwidth,figheight*2))
    ax[0].plot(xScene,np.log(sceneLikelihoods),'bd')
    ax[0].plot([np.min(xScene),np.max(xScene)],[logthres2,logthres2],'k--')
    ax[0].set_title('Likelihood test for scene data set')
    ax[0].set_xlabel('Edges')
    ax[0].set_ylabel('Log-likelihood ratio')
    plt.tight_layout()
    
    ax[1].plot(xSynthetic,np.log(syntheticLikelihoods),'rs')
    ax[1].plot([np.min(xSynthetic),np.max(xSynthetic)],[logthres2,logthres2],'k--')
    ax[1].set_title('Likelihood test for synthetic data set')
    ax[1].set_xlabel('Edges')
    ax[1].set_ylabel('Log-likelihood ratio')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print(" ** Load training images to determine an appropriate gradient threshold" \
        " for edge detection")
    tryEdgeDetection()
    sleep(3)
    print(" ** Generate histograms of the edge length in the data")
    edgeLength()
    sleep(3)
    print(" ** Evaluate the performance of the proposed test on the non-training data")
    trialEdgeLength()
    sleep(3)
    print(" ** Load all images and compute the likelihood ratios for each data set")
    edgeTrialCompare()
    sleep(3)
    print(" ** Done")
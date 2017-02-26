# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: fatadama
@brief: load images and use numpy to compute the image gradients
"""

import numpy as np
from loadImg import *
import matplotlib.pyplot as plt
from PIL import Image
from scipy import fftpack
from scipy import stats

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

    # find the discriminatorspatial
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
    im2 = im2intensity(loadSynthetic(iu))
    im = im2intensity(loadScene(iu))
    ff2 = np.fft.fft2(np.pad(im2,5,'constant',constant_values=0))
    ff = np.fft.fft2(np.pad(im,5,'constant',constant_values=0))
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
    
    # export bins values to file
    outv = np.vstack((barx,hist2,hist1)).transpose()
    print(outv.shape)
    np.savetxt('fftBins.txt',outv,delimiter=',',header='log_10(fft magnitude),synthetic pmf,scene pmf')
    
    plt.show()
    
def trialFFT():
    dat = np.genfromtxt('fftBins.txt',delimiter=',')
    print(dat)
    falsePositives = 0
    for k in range(len(SceneList)):
        io = im2intensity(loadScene(k+1))
        logff = np.log10(np.fft.fft2(im))
        np.histogram(logff,bins=dat[:,0])
        print(k,metric)
        if (metric >= xlower) and (metric <= xupper):
            # synthetic image detected
            falsePositives = falsePositives + 1
            print("False positive")
    print("%d false positives out of %d" % (falsePositives,len(SceneList)))
    pass

def tryIntensityHistogram():
    d1 = 2
    d2 = 3
    
    bins = np.arange(0,255.0001,8)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    
    fig,ax = plt.subplots(d1,d2)
    
    sceneValues = np.zeros((TrainingSize,))
    for k in range(d1):
        for j in range(d2):
            h = k*d2 + j
            im = im2intensity(loadScene(h+1))
            imhist = np.histogram(im.flatten(),bins=bins,normed=True)[0]
            # compute the diff
            m = np.sum(np.abs(np.diff(imhist)))
            sceneValues[h] = m
            ax[k,j].bar(barx,imhist,width=binwidth,color='blue')
    print('----')
        
    syntheticValues = np.zeros((TrainingSize,))
    for k in range(d1):
        for j in range(d2):
            h = k*d2 + j
            im = im2intensity(loadSynthetic(h+1))
            imhist = np.histogram(im.flatten(),bins=bins,normed=True)[0]
            # compute the diff
            m = np.sum(np.abs(np.diff(imhist)))
            sceneValues[h] = m
            ax[k,j].bar(barx+0.25*binwidth,imhist,width=0.5*binwidth,color='green')
def trainIntensityHistogram():
    bins = np.arange(0,255.0001,8)
    barx = np.diff(bins)*0.5 + bins[:-1]
    
    sceneValues = np.zeros((TrainingSize,))
    for h in range(TrainingSize):
        im = im2intensity(loadScene(h+1))
        imhist = np.histogram(im.flatten(),bins=bins,normed=True)[0]
        # normalize
        imhist = imhist.astype(float)/np.sum(imhist)
        # compute the diff
        m = np.sum(np.abs(np.diff(imhist)))
        sceneValues[h] = m
    print('----')
    syntheticValues = np.zeros((TrainingSize,))
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1))
        imhist = np.histogram(im.flatten(),bins=bins,normed=True)[0]
        # normalize
        imhist = imhist.astype(float)/np.sum(imhist)
        # compute the diff
        m = np.sum(np.abs(np.diff(imhist)))
        syntheticValues[h] = m
        
    bins2 = np.arange(0.0,1.00001,0.1)
    # TODO likelihood ratio
    lb = np.max(sceneValues)
    ub = np.min(syntheticValues)
    print("(%f,%f)" % (lb,ub))
    cutoff = np.mean((lb,ub))
    print("%f" % cutoff)
    plt.figure()
    plt.hist(sceneValues,bins2)
    plt.hist(syntheticValues,bins2,rwidth=0.5)
    plt.show()

def trialIntensityHistogram(cutoff=0.729):
    """
    Works well for case of no specular reflections only
    
    Use 0.554 for the specular reflections, works OK but not great
    """
    bins = np.arange(0,255.0001,8)
    barx = np.diff(bins)*0.5 + bins[:-1]
    
    sceneValues = np.zeros((len(SceneList),))
    for h in range(len(SceneList)):
        im = im2intensity(loadScene(h+1))
        imhist = np.histogram(im.flatten(),bins=bins)[0]
        # normalize
        imhist = imhist.astype(float)/np.sum(imhist)
        # compute the diff
        m = np.sum(np.abs(np.diff(imhist)))
        sceneValues[h] = m
    performance = np.zeros(sceneValues.shape)
    performance[sceneValues <= cutoff] = 0.0
    performance[sceneValues > cutoff] = 1.0
    syntheticValues = np.zeros((99,))
    for h in range(99):
        im = im2intensity(loadSynthetic(h+1))
        imhist = np.histogram(im.flatten(),bins=bins)[0]
        # normalize
        imhist = imhist.astype(float)/np.sum(imhist)
        # compute the diff
        m = np.sum(np.abs(np.diff(imhist)))
        syntheticValues[h] = m
    performance2 = np.zeros(syntheticValues.shape)
    performance2[syntheticValues <= cutoff] = 0.0
    performance2[syntheticValues > cutoff] = 1.0
    
    print("False positive rate: %d of %d" % (np.sum(performance),len(performance)))
    print("True detection rate: %d of %d" % (np.sum(performance2),len(performance2)))
    '''
    plt.figure()
    plt.hist(sceneValues)
    plt.hist(syntheticValues,rwidth=0.5)
    ''' 
    pass

def test1DFFT():
    """ Compute the 1D FFT of test images
    """
    #target size
    #sz = [550,970]
    sz = [275,485]
    axis = 1
    d1 = 3
    d2 = 3
    bins = np.arange(-3,6.0001,0.25)
    barx = np.diff(bins)*0.5 + bins[:-1]
    fig,ax = plt.subplots(d1,d2)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            print(h)
            # load iamge and pad with zeros
            im = np.pad(im2intensity(loadSynthetic(h+1,sz=sz)),5,'constant',constant_values=0)
            # 1D FFT
            #ff = np.fft.fft(im,axis=axis)
            ff = np.fft.fft2(im)
            print(np.log10(np.abs(ff)).shape)
            #ax[k,j].hist(np.log10(np.abs(ff)).flatte
            #ax[k,j].set_xlim((bins[0],bins[-1]))n(),bins,normed=True)
            ax[k,j].imshow(np.log10(np.abs(ff)))
            ax[k,j].set_title('Synthetic %d' % (h+1))
    fig,ax = plt.subplots(d1,d2)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            print(h)
            # load iamge and pad with zeros
            im = np.pad(im2intensity(loadScene(h+1,sz=sz)),5,'constant',constant_values=0)
            # 1D FFT
            #ff = np.fft.fft(im,axis=axis)
            ff = np.fft.fft2(im)
            print(np.log10(np.abs(ff)).shape)
            #ax[k,j].hist(np.log10(np.abs(ff)).flatten(),bins,normed=True)
            #ax[k,j].set_xlim((bins[0],bins[-1]))
            ax[k,j].imshow(np.log10(np.abs(ff)))
            ax[k,j].set_title('Scene %d' % (h+1))
    plt.show()
    
def train2DFFT():
    """
    Compute the mean 2D FFTs and plot/save
    
    
    """
    sz = [275,485]
    pad = 5
    meanScene = np.zeros((sz[0]+2*pad,sz[1]+2*pad))
    for k in range(TrainingSize):
        print(k)
        # scenes
        im = np.pad(im2intensity(loadScene(k+1,sz=sz)),pad,'constant',constant_values=0)
        # log10 abs fft2
        ff = np.log10(np.abs(np.fft.fft2(im)))
        # take mean
        meanScene = meanScene + 1.0/TrainingSize*ff.astype(float)
    
    meanSynthetic = np.zeros((sz[0]+2*pad,sz[1]+2*pad))
    for k in range(TrainingSize):
        print(k)
        # scenes
        im = np.pad(im2intensity(loadSynthetic(k+1,sz=sz)),pad,'constant',constant_values=0)
        # log10 abs fft2
        ff = np.log10(np.abs(np.fft.fft2(im)))
        # take mean
        meanSynthetic = meanSynthetic + 1.0/TrainingSize*ff.astype(float)
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(meanScene)
    ax[1].imshow(meanSynthetic)
    plt.show()
    return
    
def azimuthalAvg(im):
    # find center
    ct = (int(im.shape[0]/2),int(im.shape[1]/2))
    # radius to center
    x,y = np.indices(im.shape)
    R = np.sqrt(np.power(x-ct[0],2.0)+np.power(y-ct[1],2.0)).astype(int)
    # find isolines of R
    N = np.max(R)-np.min(R)+1
    azimuthalMean = np.zeros((N,))
    count = 0
    for k in range(np.min(R),np.max(R)+1):
        # find
        ix = np.where(R.flat==k)[0]
        if not(ix.shape[0] == 0):
            # compute the average
            azimuthalMean[count] = np.mean(im.flat[ix])
        print(count,azimuthalMean[count])
        count = count+1
    return azimuthalMean
    
def azimuthalAvgFast(image, center=None):
    # Calculate the indices from the image
    y, x = np.indices((image.shape[0],image.shape[1]))
    if not center:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    if len(image.shape) > 2:
        N = np.max(r.flat)-1
        radial_prof = np.zeros((N,image.shape[2]))
        for k in range(image.shape[2]):
            im = image[:,:,k]
            # Get sorted radii
            ind = np.argsort(r.flat)
            r_sorted = r.flat[ind]
            i_sorted = im.flat[ind]
            # Get the integer part of the radii (bin size = 1)
            r_int = r_sorted.astype(int)
            # Find all pixels that fall within each radial bin.
            # Assumes all radii represented
            deltar = r_int[1:] - r_int[:-1]
            # location of changed radius
            rind = np.where(deltar)[0]
            # number of radius bin
            nr = rind[1:] - rind[:-1]
            # Cumulative sum to figure out sums for each radius bin
            csim = np.cumsum(i_sorted, dtype=float)
            tbin = csim[rind[1:]] - csim[rind[:-1]]
            radial_prof[:,k] = tbin / nr
    else:
        # Get sorted radii
        ind = np.argsort(r.flat)
        r_sorted = r.flat[ind]
        i_sorted = image.flat[ind]
        # Get the integer part of the radii (bin size = 1)
        r_int = r_sorted.astype(int)
        # Find all pixels that fall within each radial bin.
        # Assumes all radii represented
        deltar = r_int[1:] - r_int[:-1]
        # location of changed radius
        rind = np.where(deltar)[0]
        # number of radius bin
        nr = rind[1:] - rind[:-1]
        # Cumulative sum to figure out sums for each radius bin
        csim = np.cumsum(i_sorted, dtype=float)
        tbin = csim[rind[1:]] - csim[rind[:-1]]
        radial_prof = tbin / nr
    return radial_prof
    
def testFFTazimuthalAvg():
    """ Plot d1 x d2 array of the azimuthal average FFTs for the first 
    d1*d2 scenes and synthetic images
    """
    sz = [550,970]
    d1 = 2
    d2 = 3
    #fig,ax = plt.subplots(d1,d2)
    fig,ax = plt.subplots(1,2,sharey=True)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            im = im2intensity(loadSynthetic(h+1,sz=sz))
            F1 = fftpack.fft2(im)
            ff = np.power(np.abs(fftpack.fftshift(F1)),2.0)
            avg = azimuthalAvgFast(ff)
            print(avg.shape)
            
            x_values = np.array(range(len(avg))).astype(float)*1.0/float(len(avg))
            y_values = ax[0].semilogy(x_values,avg)[0].get_ydata()
            ax[0].set_title('Synthetic %d' % (h+1))
            '''
            if d1 == 1:
                y_values = ax[j].semilogy(avg)[0].get_ydata()
                ax[j].set_title('Synthetic %d' % (h+1))
            else:
                y_values = ax[k,j].semilogy(avg)[0].get_ydata()
                ax[k,j].set_title('Synthetic %d' % (h+1))
            '''
            print("%7.2g,%7.2g,%7.2g" % (np.max(y_values),np.mean(y_values[:100]),np.mean(y_values[100:])))
            #print("%7.2g" % np.sum(y_values[1:]-y_values[:-1]))
            #print("%7.2g" %  np.sum([i-j for i, j in zip(y_values[:-1],y_values[1:])][100:len(y_values)]) )
    print('----')
    #fig,ax = plt.subplots(d1,d2)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            im = im2intensity(loadScene(h+1,sz=sz))
            F1 = fftpack.fft2(im)
            ff = np.power(np.abs(fftpack.fftshift(F1)),2.0)
            avg = azimuthalAvgFast(ff)
            print(avg.shape)
            
            x_values = np.array(range(len(avg))).astype(float)*1.0/float(len(avg))
            y_values = ax[1].semilogy(x_values,avg)[0].get_ydata()
            ax[1].set_title('Scene %d' % (h+1))
            '''
            if d1 == 1:
                y_values = ax[j].semilogy(avg)[0].get_ydata()
                ax[j].set_title('Scene %d' % (h+1))
            else:
                y_values = ax[k,j].semilogy(avg)[0].get_ydata()
                ax[k,j].set_title('Scene %d' % (h+1))
            '''
            print("%7.2g,%7.2g,%7.2g" % (np.max(y_values),np.mean(y_values[:100]),np.mean(y_values[100:])))
            #print("%7.2g" % np.sum(y_values[1:]-y_values[:-1]))
            #print("%7.2g" %  np.sum([i-j for i, j in zip(y_values[:-1],y_values[1:])][100:len(y_values)]) )
    plt.show()
    
def testFFTPeakValue():
    sz = [550,970]
    valSynthetic = np.zeros((TrainingSize,2))
    valScene = np.zeros((TrainingSize,2))
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        F1 = fftpack.fft2(im)
        ff = np.power(np.abs(fftpack.fftshift(F1)),2.0)
        avg = azimuthalAvgFast(ff)
        valSynthetic[h,0] = np.max(avg)
        valSynthetic[h,1] = np.mean(avg[-100:-1])
        print(h,valSynthetic[h])
    print('----')
    for h in range(TrainingSize):
        im = im2intensity(loadScene(h+1,sz=sz))
        F1 = fftpack.fft2(im)
        ff = np.power(np.abs(fftpack.fftshift(F1)),2.0)
        avg = azimuthalAvgFast(ff)
        valScene[h,0] = np.max(avg)
        valScene[h,1] = np.mean(avg[-100:-1])
        print(h,valScene[h])
    # bins for peak vaue histogram
    bins = np.arange(10,20.0001,0.25)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    sceneBar = np.histogram(np.log10(valScene[:,0]),bins,normed=True)[0]
    syntheticBar = np.histogram(np.log10(valSynthetic[:,0]),bins,normed=True)[0]
    # peak value histogram
    fig = plt.figure()
    plt.bar(barx,sceneBar,width=binwidth)
    plt.bar(barx+0.25*binwidth,syntheticBar,width=0.5*binwidth,color='red')
    
    # first 100 mean histogram
    sceneBar1 = np.histogram(np.log10(valScene[:,1]),bins,normed=True)[0]
    syntheticBar1 = np.histogram(np.log10(valSynthetic[:,1]),bins,normed=True)[0]
    fig = plt.figure()
    plt.bar(barx,sceneBar1,width=binwidth)
    plt.bar(barx+0.25*binwidth,syntheticBar1,width=0.5*binwidth,color='red')
    
    # show figs
    plt.show()
    pass
    
def simpleFFTazimuthalAvg():
    """ Run one case at a time, plot the 2D FFT
    """
    # loading function to use
    func = loadSynthetic
    # image number to look at
    imin = 2
    # load image
    im = im2intensity(func(imin))
    # stuff
    F1 = fftpack.fft2(im)
    ff = np.power(np.abs(fftpack.fftshift(F1)),2.0)
    avg = azimuthalAvgFast(ff)
    
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(im,cmap='gray')    
    ax[1].imshow(np.log10(ff))
    ax[2].plot(np.log10(avg))
    ax[2].set_ylabel(r'$\log_{10}$(radial FFT)')
    ax[2].grid()
    
    plt.show()
    
def tryEdgeDetection():
    """ Load a scene and a synthetic image and plot the intensity gradients for
    each
    """
    bins = np.arange(0.0,128.0001,4)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    
    sz = [550,970]
    
    fig,ax = plt.subplots(1,1)
    
    valSynthetic = np.zeros((TrainingSize,sz[0],sz[1]))
    valScene = np.zeros((TrainingSize,sz[0],sz[1]))
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        valSynthetic[h,:,:] = np.gradient(im)[0]
    print('----')
    for h in range(TrainingSize):        
        im = im2intensity(loadScene(h+1,sz=sz))
        valScene[h,:,:] = np.gradient(im)[0]
    
    barSynthetic = np.histogram(valSynthetic,bins,normed=True)[0]
    barScene = np.histogram(valScene,bins,normed=True)[0]
    
    print(barx)
    
    ax.bar(barx,barScene,width=binwidth,color='blue')
    ax.bar(barx+0.25*binwidth,barSynthetic,width=0.5*binwidth,color='green')
        
    plt.show()
    
def edgeLength():

    thres = 4.0    
    
    bins = np.linspace(0.0,400000,20)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    
    sz = [550,970]
    
    fig,ax = plt.subplots(2,1)
    
    valSynthetic = np.zeros((TrainingSize,))
    valScene = np.zeros((TrainingSize,))
    for h in range(TrainingSize):
        im = im2intensity(loadSynthetic(h+1,sz=sz))
        gra = np.gradient(im)[0]
        valSynthetic[h] = len(np.where(np.abs(gra) > thres)[0])
    print('----')
    for h in range(TrainingSize):        
        im = im2intensity(loadScene(h+1,sz=sz))
        gra = np.gradient(im)[0]
        valScene[h] = len(np.where(np.abs(gra) > thres)[0])
    
    scenefit = stats.cauchy.fit(valScene)
    syntheticfit = stats.norm.fit(valSynthetic)
    print( scenefit )
    print( syntheticfit )
    print( stats.norm.fit(valSynthetic) )
    
    bins = np.linspace(0.0,400000,10)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))
    barScene = np.histogram(valScene,bins,normed=True)[0]
    
    ax[0].bar(bins[:-1],barScene,width=binwidth,color='blue')
    #ax[0].hist(valScene,rwidth=1,normed=True)
    ax[0].plot(barx,stats.cauchy.pdf(barx,loc=scenefit[0],scale=scenefit[1]),'k-x')
    ax[0].set_title('Edge length histograms')
    
    bins = np.linspace(0,100000,10)
    barx = np.diff(bins)*0.5 + bins[:-1]
    binwidth = np.mean(np.diff(bins))    
    barSynthetic = np.histogram(valSynthetic,bins,normed=True)[0]
    ax[1].bar(bins[:-1],barSynthetic,width=binwidth,color='green')
    ax[1].plot(barx,stats.norm.pdf(barx,loc=syntheticfit[0],scale=syntheticfit[1]),'k-x')
    #ax[1].hist(valSynthetic,rwidth=1,normed=True)
        
    plt.show()
    
def trialEdgeLength():
    # null hypothesis: normal with scale=38192,loc = 8817
    # alternate hypothesis: cauchy with 153595, loc = 45142

    thres = 4.0    
        
    sz = [550,970]
        
    testingSize0 = 99-TrainingSize
    testingSize1 = len(SceneList)-TrainingSize
    
    valSynthetic = np.zeros((testingSize0,))
    valScene = np.zeros((TrainingSize,))
    for h in range(testingSize0):
        im = im2intensity(loadSynthetic(h+TrainingSize,sz=sz))
        gra = np.gradient(im)[0]
        valSynthetic[h] = len(np.where(np.abs(gra) > thres)[0])
    print('----')
    for h in range(testingSize1):
        im = im2intensity(loadScene(h+TrainingSize,sz=sz))
        gra = np.gradient(im)[0]
        valScene[h] = len(np.where(np.abs(gra) > thres)[0])
    print(valSynthetic)
    print(valScene)
    
    p00 = stats.norm.pdf(valSynthetic, scale=38192,loc=8817)
    p10 = stats.cauchy.pdf(valSynthetic,153595,45142)
    p01 = stats.norm.pdf(valScene,scale=38192,loc=8817)
    p11 = stats.cauchy.pdf(valScene,153595,45142)
    
    l0 = p10/p00
    l1 = p11/p01
    
    w0 = np.where(l0 > 1)[0]
    w1 = np.where(l1 < 1)[0]
    
    print("%d false postives out of %d" % (len(w0),len(l0)))
    print("%d false negatives out of %d" % (len(w1),len(l1)))
    
    fig,ax = plt.subplots(2,1)
    ax[0].plot(valSynthetic,p00,'gx')
    ax[0].plot(valSynthetic,p10,'bd')
    
    ax[1].plot(valScene,p01,'gx')
    ax[1].plot(valScene,p11,'bd')
    
    plt.show()

    # load scenes
    
def plotGradients():
    sz = [550,970]
    d1 = 1
    d2 = 2
    fig,ax = plt.subplots(d1,d2)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            im = np.gradient(im2intensity(loadSynthetic(h+1,sz=sz)))[0]
            if d1 == 1:
                ax[j].imshow(im)
                ax[j].set_title('Synthetic %d' % (h+1))
            else:
                ax[k,j].imshow(im)
                ax[k,j].set_title('Synthetic %d' % (h+1))
    print('----')
    fig,ax = plt.subplots(d1,d2)
    for k in range(d1):
        for j in range(d2):
            h = j + k*d2
            im = np.gradient(im2intensity(loadScene(h+1,sz=sz)))[0]            
            if d1 == 1:
                ax[j].imshow(im)
                ax[j].set_title('Scene %d' % (h+1))
            else:
                ax[k,j].imshow(im)
                ax[k,j].set_title('Scene %d' % (h+1))
    plt.show()
    
if __name__ == '__main__':
    #testMeanGradient()
    #testPlotFFT(10)
    #testFFT()
    #trialFFT()
    #trialMeanGradient()
    #edgeLength()
    trialEdgeLength()
    #plotGradients()
    #trainIntensityHistogram()
    #trialIntensityHistogram(.528)
    #test1DFFT()
    #train2DFFT()
    #testFFTazimuthalAvg()
    #testFFTPeakValue()
    pass
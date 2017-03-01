# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 16:05:54 2017

@author: fatadama
"""

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
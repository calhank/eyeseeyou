# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:39:38 2015

@author: Daniel
"""

### Imaging functions for emphasizing features

# NOTE:   These functions return copies of the original array
# so as not to destroy the original

import numpy as np
from scipy import ndimage

def hipass(image,gaussrad,dilrat):
    # 
    # A derivative of gaussian (suggested radii of 0.5) to get edges
    # followed by a morphological dilate to widen the effect and pick up extra pixels on edges
    # This can be used as a mask for 

    timg = np.copy(image).reshape(96,96)
    
    gradim = ndimage.gaussian_gradient_magnitude(timg,gaussrad)
    
    digradim= ndimage.morphology.grey_dilation(gradim,dilrat)
    
    return digradim.flatten()
    

def thresh(image,thr):
    
    timg = np.copy(image).reshape(96,96)
    
    timg[timg<thr]=0
    
    timg[timg>=thr]=255
    
    return timg.flatten()
    
def neg(image):
    
    timg =np.copy(image).reshape(96,96)
    
    timg = 255-timg
    
    return timg.flatten()


def binarize(image):
    
    # approximate half the pixels set to 255 , half to 0
    
    timg = np.copy(image)
    
    thr = np.median(timg)
    
    return thresh(timg,thr).flatten()


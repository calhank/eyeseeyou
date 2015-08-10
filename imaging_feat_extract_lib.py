# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 00:39:38 2015

@author: Daniel
"""

### Imaging functions for emphasizing features

from scipy import ndimage

def hipass(image,gaussrad,dilrat):
    # 
    # A derivative of gaussian (suggested radii of 0.5) to get edges
    # followed by a morphological dilate to widen the effect and pick up extra pixels on edges
    # This can be used as a mask for 

    timg = image.reshape(96,96)
    
    gradim = ndimage.gaussian_gradient_magnitude(timg,.5)
    
    digradim= ndimage.morphology.grey_dilation(gradim,1)
    
    return digradim.flatten()
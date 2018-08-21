#!/usr/bin/env python3

"""
Implementation of RAISR in Python by Jalali-Lab

[RAISR](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7744595) (Rapid and Accurate Image Super
Resolution) is an image processing algorithm published by Google Research in 2016. With sufficient
trainingData, consisting of low and high resolution image pairs, RAISR algorithm tries to learn
a set of filters which can be applied to an input image that is not in the training set,
to produce a higher resolution version of it. The source code released here is the Jalali-Lab
implementation of the RAISR algorithm in Python 3.x. The implementation presented here achieved
performance results that are comparable to that presented in Google's research paper
(with ± 0.1 dB in PSNR).

Just-in-time (JIT) compilation employing JIT numba is used to speed up the Python code. A very
parallelized Python code employing multi-processing capabilities is used to speed up the testing
process. The code has been tested on GNU/Linux and Mac OS X 10.13.2 platforms.

Author: Sifeng He, Jalali-Lab, Department of Electrical and Computer Engineering, UCLA

Copyright
---------
RAISR is developed in Jalali Lab at University of California, Los Angeles (UCLA).
More information about the technique can be found in our group website: http://www.photonics.ucla.edu


"""

import numpy as np
import numba as nb
import os
import cv2
import warnings
import pickle
import time
from math import floor
from skimage.measure import compare_psnr
from scipy.misc import imresize
from Functions import *
import matplotlib.image as mpimg
from multiprocessing import Pool

warnings.filterwarnings('ignore')

testLRPath = 'testDataLR'
testHRPath = 'testDataHR'

R = 2  # Upscaling factor=2 R = [ 2 3 4 ]
patchSize = 11  # Pacth Size=11
gradientSize = 9
Qangle = 24  # Quantization factor of angle =24
Qstrength = 3  # Quantization factor of strength =3
Qcoherence = 3  # Quantization factor of coherence =3

with open("Filter/filter"+str(R), "rb") as fp:
    h = pickle.load(fp)

with open("Filter/Qfactor_str"+str(R), "rb") as sp:
    stre = pickle.load(sp)

with open("Filter/Qfactor_coh"+str(R), "rb") as cp:
    cohe = pickle.load(cp)

# Get HR and LR test data
filelistLR = make_dataset(testLRPath)
filelistHR = make_dataset(testHRPath)
# Zip together HR and LR lists
# Index 0: LR
# Index 1: HR
filelist = np.array((filelistLR, filelistHR))

wGaussian = Gaussian2d([patchSize, patchSize], 2)
wGaussian = wGaussian/wGaussian.max()
wGaussian = np.diag(wGaussian.ravel())
imagecount = 1
patchMargin = floor(patchSize/2)
psnrRAISR = []
psnrBicubic = []
psnrBlending = []

def TestProcess(i):
    if (i < iteration - 1):
        offset_i = offset[i * batch:i * batch + batch]
        offset2_i = offset2[i * batch:i * batch + batch]
        grid = np.tile(gridon[..., None], [1, 1, batch]) + np.tile(offset_i, [patchSize, patchSize, 1])
    else:
        offset_i = offset[i * batch:imHR.size]
        offset2_i = offset2[i * batch:imHR.size]
        grid = np.tile(gridon[..., None], [1, 1, imHR.size - (iteration - 1) * batch]) + np.tile(offset_i,[patchSize, patchSize,1])
    f = im_LR.ravel()[grid]
    # Use gradient statistics to find theta, lamda, and miu 
    # theta: angle
    # lamda: strength 
    # miu: coherence
    # theta, lamda, miu used to access hash table for appropriate filter 
    # Gradient 
    gx = im_GX.ravel()[grid]
    gy = im_GY.ravel()[grid]
    gx = gx.reshape((1, patchSize * patchSize, gx.shape[2]))
    gy = gy.reshape((1, patchSize * patchSize, gy.shape[2]))
    G = np.vstack((gx, gy))
    g1 = np.transpose(G, (2, 0, 1))
    g2 = np.transpose(G, (2, 1, 0))
    x = Gaussian_Mul(g1, g2, wGaussian)
    w, v = np.linalg.eig(x)
    idx = (-w).argsort()
    w = w[np.arange(np.shape(w)[0])[:, np.newaxis], idx]
    v = v[np.arange(np.shape(v)[0])[:, np.newaxis, np.newaxis], np.arange(np.shape(v)[1])[np.newaxis, :, np.newaxis], idx[:,np.newaxis,:]]
    # Gradient angle 
    thelta = np.arctan(v[:, 1, 0] / v[:, 0, 0])
    thelta[thelta < 0] = thelta[thelta < 0] + pi
    thelta = np.floor(thelta / (pi / Qangle))
    thelta[thelta > Qangle - 1] = Qangle - 1
    thelta[thelta < 0] = 0
    # Gradient strength 
    lamda = w[:, 0]
    # Gradient coherence 
    u = (np.sqrt(w[:, 0]) - np.sqrt(w[:, 1])) / (np.sqrt(w[:, 0]) + np.sqrt(w[:, 1]) + 0.00000000000000001)
    lamda = np.searchsorted(stre, lamda)
    u = np.searchsorted(cohe, u)
    # Calculate j: Quantize 
    j = thelta * Qstrength * Qcoherence + lamda * Qcoherence + u
    j = j.astype('int')
    offset2_i = np.unravel_index(offset2_i, (H, W))
    t = ((offset2_i[0] - patchMargin) % R) * R + ((offset2_i[1] - patchMargin) % R)
    # Obtain appropriate filter 
    filtertj = h[t, j]
    filtertj = filtertj[:, :, np.newaxis]
    patch = f.reshape((1, patchSize * patchSize, gx.shape[2]))
    patch = np.transpose(patch, (2, 0, 1))
    # Apply filter to patch 
    result = np.matmul(patch, filtertj)
    return result


print('Begin to process images:')
for image in range(0, filelist.shape[1]):
    print('\r', end='')
    print('' * 60, end='')
    print('\r Processing ' + str(imagecount) + '/' + str(filelist.shape[1]) + ' LR image (' + filelist[0][image] + ') HR image (' + filelist[1][image] + ')')
        
    # Read in images
    # Low Resolution 
    im_uint8LR = cv2.imread(filelist[0][image])
    im_mpLR = mpimg.imread(filelist[0][image])
    # High Resolution: Fourier ground truth 
    im_uint8HR = cv2.imread(filelist[1][image])
    im_mpHR = mpimg.imread(filelist[1][image])
    # Prepare Images
    # LR images 
    if len(im_mpLR.shape) == 2:
        im_uint8LR = im_uint8LR[:,:,0]
    im_uint8LR = modcrop(im_uint8LR, R)
    if len(im_uint8LR.shape) > 2:
        im_ycbcrLR = BGR2YCbCr(im_uint8LR)
        imLR = im_ycbcrLR[:, :, 0]
    else:
        imLR = im_uint8LR
    # HR images 
    if len(im_mpHR.shape) == 2:
        im_uint8HR = im_uint8HR[:,:,0]
    im_uint8HR = modcrop(im_uint8HR, R)
    if len(im_uint8HR.shape) > 2:
        im_ycbcrHR = BGR2YCbCr(im_uint8HR)
        imHR = im_ycbcrHR[:, :, 0]
    else:
        imHR = im_uint8HR
    # Scale LR input image to a 0.0 to 1.0 range
    im_doubleLR = im2double(imLR)
    # Retrieve the width and height of the HR image
    H, W = imHR.shape
    # Generate region
    region = (slice(patchMargin, H - patchMargin), slice(patchMargin, W - patchMargin))
    start = time.time()
    # Bicubic interpolation from LR image to HR image
    im_bicubic = imresize(im_doubleLR, (H, W), interp='bicubic', mode='F')
    # Cast image into a 64-bit float
    im_bicubic = im_bicubic.astype('float64')
    # Clip image values to a 0.0 to 1.0 range
    im_bicubic = np.clip(im_bicubic, 0, 1)
    # Save bicubic interpolated image into im_LR
    im_LR = np.zeros((H+patchSize-1,W+patchSize-1))
    im_LR[(patchMargin):(H+patchMargin),(patchMargin):(W+patchMargin)] = im_bicubic
    
    # Preperation of enhancement using filters
    # Initialize filtered image with matrix of zeros 
    im_result = np.zeros((H, W))
    # Generate gradients of upscaled LR image 
    im_GX, im_GY = np.gradient(im_LR)
    # Generate matrix of indices for upscaled LR images 
    index = np.array(range(im_LR.size)).reshape(im_LR.shape)
    # Flatten indices of indices matrix 
    offset = np.array(index[0:H, 0:W].ravel())
    # Get array of indices of HR image 
    offset2 = np.array(range(imHR.size))
    # Get patch matrix 
    gridon = index[0:patchSize, 0:patchSize]
    # Define batch number
    batch = 2000
    # Calculate number of iterations 
    iteration = ceil(imHR.size / batch + 0.000000000000001)
    # Set result to empty array 
    im_result = np.array([])
    # Define i to be a range from 0 to number of iterations 
    i = range(iteration)
    # Initialize threading based interface 
    p = Pool()
    # Enhancement using filters
    im_in = p.map(TestProcess, i)
    # Combine pieces of enhancement 
    for i in range(iteration):
        im_result = np.append(im_result, im_in[i])
    # Reshape filter enhanced 
    im_result = im_result.reshape(H, W)
    # Clip filter enhanced 
    im_result = np.clip(im_result, 0, 1)
    # Calculate elapsed time
    end = time.time()
    print(end - start)
    # Blend filter enhanced and plain interpolated images
    im_blending = Blending2(im_bicubic, im_result)
    # im_blending = Backprojection(imL, im_blending, 50) #Optional: Backprojection, which can slightly improve PSNR, especilly for large upscaling factor.
    im_blending = np.clip(im_blending, 0, 1)
    
    # Generate results
    '''
    if len(im_uint8HR.shape) > 2:
        result_ycbcr = np.zeros((H, W, 3))
        result_ycbcr[:, :, 1:3] = im_ycbcrHR[:, :, 1:3]
        result_ycbcr[:, :, 0] = im_blending * 255
        result_ycbcr = result_ycbcr[region].astype('uint8')
        result_RAISR = YCbCr2BGR(result_ycbcr)
    else:
        result_RAISR = im_result[region] * 255
        result_RAISR = result_RAISR.astype('uint8')
    '''
    # Filtered result 
    im_result = im_result*255
    im_result = np.rint(im_result).astype('uint8')
    # Bicubic result 
    im_bicubic = im_bicubic*255
    im_bicubic = np.rint(im_bicubic).astype('uint8')
    # Blended result 
    im_blending = im_blending * 255
    im_blending = np.rint(im_blending).astype('uint8')
    # Measure reconstruction quality
    # Calculate PSNR values 
    PSNR_bicubic = compare_psnr(imHR[region], im_bicubic[region])
    PSNR_result = compare_psnr(imHR[region], im_result[region])
    PSNR_blending = compare_psnr(imHR[region], im_blending[region])
    PSNR_blending = max(PSNR_result, PSNR_blending)
    # Save RAISR reconstruction 
    createFolder('./results/')
    #cv2.imwrite('results/' + os.path.splitext(os.path.basename(filelist[0][image]))[0] + '_result.bmp', result_RAISR)
    cv2.imwrite('results/' + os.path.splitext(os.path.basename(filelist[0][image]))[0] + '_bicubic.bmp', im_bicubic[region])
    cv2.imwrite('results/' + os.path.splitext(os.path.basename(filelist[0][image]))[0] + '_result.bmp', im_result[region])
    cv2.imwrite('results/' + os.path.splitext(os.path.basename(filelist[0][image]))[0] + '_blended.bmp', im_blending[region])
    psnrRAISR.append(PSNR_result)
    psnrBicubic.append(PSNR_bicubic)
    psnrBlending.append(PSNR_blending)

    imagecount += 1

# Calculate mean of psnr values 
RAISR_psnrmean = np.array(psnrBlending).mean()
Bicubic_psnrmean = np.array(psnrBicubic).mean()


print('\r', end='')
print('' * 60, end='')
print('\r RAISR PSNR of '+ testLRPath +' is ' + str(RAISR_psnrmean))
print('\r Bicubic PSNR of '+ testLRPath +' is ' + str(Bicubic_psnrmean))

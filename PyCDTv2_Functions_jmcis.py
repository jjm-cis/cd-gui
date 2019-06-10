#GG
#Author: Jobin J Mathew
#All copyrights reserved to the author and Chester F. Carlson Center for Imaging Science (CIS), at Rochester Institute
#Technology (RIT), Rochester, NY.

import cv2
import numpy as np
import time
from PIL import Image
from spectral import *


########################################################################################################################
def read_envi_img(file_name_1):

    img_read_0 = envi.open(file_name_1)
    img_open_0 = img_read_0.open_memmap(writeable=True)

    [height, width, dim] = img_read_0.shape

    # ENVI to image array
    img_multi_0_test = np.zeros((height, width, dim), dtype=np.float32)

    for i_dim0 in range(0, dim):
        ####
        # Contrast streching and histogram equalization
        img_multi_0_test[:, :, i_dim0] = img_open_0[:, :, i_dim0].copy()
        var_min_1 = img_multi_0_test[:, :, i_dim0].min()
        var_max_1 = img_multi_0_test[:, :, i_dim0].max()
        min_out = 0
        var_contr_fact_1 = np.divide(255 - min_out, var_max_1 - var_min_1) + min_out
        img_multi_0_test[:, :, i_dim0] = np.multiply((img_multi_0_test[:, :, i_dim0] - var_min_1), var_contr_fact_1)
        img_multi_0_test[:, :, i_dim0] = cv2.equalizeHist(img_multi_0_test[:, :, i_dim0].astype(np.uint8))
        ####

    img_multi_0 = img_multi_0_test.astype(np.uint8)

    return img_multi_0
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------


########################################################################################################################
def add_noise(img_multi, SNR):

    ### Parameters ###
    # SNR = 5000
    mean_gaus = 0
    sigma_gaus = 1

    (height, width, dim) = img_multi.shape
    N_gaus = (np.random.normal(mean_gaus, sigma_gaus, (height,width))).astype(np.uint8)
    for i_dim in range(0,dim):
        x0_img = (img_multi[:, :, i_dim]).astype(np.float64)
        sigma_const_1 = x0_img/SNR
        img_multi[:, :, i_dim] = (x0_img + N_gaus*sigma_const_1).astype(np.uint8)

    return img_multi
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
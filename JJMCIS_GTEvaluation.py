#GG
#Author: Jobin J Mathew
#All copyrights reserved to the author and DIRS group at Chester F. Carlson Center for Imaging Science (CIS), at Rochester Institute
#Technology (RIT), Rochester, NY.


from PIL import Image
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from scipy.integrate import simps
import time

def gt_eval(output_image, file_name_3,Tile_size,*args):

    print('-----------------------------------------------------------------')

    [height,width] = output_image.shape
    output_image = output_image.astype(np.uint8)

    if len(args)>0:
        xy_startend = args[0]
    else:
        xy_startend = [0,0,height,width]

    # Ground truth/Material IDs images
    im_0 = Image.open(file_name_3)
    im_diff_thresh = np.array(im_0)

    print("Processing images for tilesize = %d ..."%(Tile_size))

    hit_list = []
    fa_list = []
    fa_list.append(1)
    hit_list.append(1)

    for t in range(0,254):
        ret, thresh1 = cv2.threshold(output_image, t, 255, cv2.THRESH_BINARY)

        fa_list.append(0)
        hit_list.append(0)
        tot_tile = 0
        tot_present = 0
        tot_absent = 0
        hit_count = 0
        fa_count = 0
        i_temp = 0
        j_temp = 0

        if Tile_size > 1:
            for i in range(xy_startend[0],xy_startend[2],Tile_size):
                for j in range(xy_startend[1],xy_startend[3],Tile_size):

                    imdiff_tile = im_diff_thresh[i:i+Tile_size,j:j+Tile_size]
                    im_out_tile = thresh1[i_temp:i_temp+Tile_size,j_temp:j_temp+Tile_size]

                    tot_tile = tot_tile + 1
                    j_temp += Tile_size

                    sum_tile = np.sum(imdiff_tile)
                    sum_res_tile = np.sum(im_out_tile)

                    # TOTAL counts of hits, misses, fa's and cr's
                    if sum_tile>0:
                        tot_present = tot_present + 1
                    else:
                        tot_absent = tot_absent + 1

                    # COUNTS
                    if (sum_tile>0) and (sum_res_tile>0):
                        hit_count = hit_count+1
                    elif (sum_tile==0) and (sum_res_tile>0):
                        fa_count = fa_count + 1

                i_temp += Tile_size
                j_temp = 0
        else:
            hit_image = cv2.bitwise_and(im_diff_thresh,thresh1)
            # cv2.imshow('hit_image',hit_image)
            # cv2.waitKey()
            fa_image = cv2.bitwise_and(cv2.bitwise_not(im_diff_thresh),thresh1)
            tot_present = cv2.countNonZero(im_diff_thresh)
            tot_absent = cv2.countNonZero(cv2.bitwise_not(im_diff_thresh))
            hit_count = cv2.countNonZero(hit_image)
            fa_count = cv2.countNonZero(fa_image)
        # HIT & FA percentages
        hit_list[t+1] = hit_count/tot_present
        fa_list[t+1] = fa_count/tot_absent

    fa_list.append(0)
    hit_list.append(0)
    AUROC = np.trapz(hit_list,x=fa_list)
    print("Area under ROC   -----  Tile size N = %d     ----- AUROC = %.2f"%(Tile_size, AUROC))

    return fa_list, hit_list, AUROC

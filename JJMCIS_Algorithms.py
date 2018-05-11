#GG
#Author: Jobin J Mathew
#All copyrights reserved to the author and DIRS group at Chester F. Carlson Center for Imaging Science (CIS), at Rochester Institute
#Technology (RIT), Rochester, NY.

# FUNCTIONS FOR CHANGE DETECTION

# import matlab.engine
import cv2
import numpy as np
import time
import sys
import GT_Evaluation_v1 as GTEval
from PIL import Image
import matplotlib.pyplot as plt

########################################################################################################################
def pdtl_cd(img_multi_0, img_multi_1, Tile_size):

    [height, width, dim] = img_multi_0.shape

    tile_size = Tile_size
    TILE_SQR = tile_size*tile_size

    im0_temp1 = np.zeros((height, width), dtype=np.float64)
    h1 = height/100  # For status percentage

    # ---------------- Processing tile-by-tile
    for i in range(0, height, tile_size):

        # Show progress percentage
        sys.stdout.write('\r')
        sys.stdout.write("Status:    %d%%" % ((i + 1) / h1))
        sys.stdout.flush()

        if (i + tile_size) > height:
            break

        for j in range(0, width, tile_size):

            if (j + tile_size) > width:
                break

            # ---------------- Select tiles from Image 0 and Image 1
            im0_tile = img_multi_0[i:i + tile_size, j:j + tile_size, :]  # Size: tile_size x tile_size x dim
            im1_tile = img_multi_1[i:i + tile_size, j:j + tile_size, :]

            thresh_tail = 0.0001/100
            PDTL_0 = pdtl_tile(im0_tile)
            PDTL_1 = pdtl_tile(im1_tile)
            PDTL = abs(PDTL_0 - PDTL_1)

            nan_check = np.isnan(PDTL)
            if nan_check == True:
                PDTL = 0

            # if PDTL<thresh_tail:
            #     PDTL = 0

            im0_temp1[i:i + tile_size, j:j + tile_size] = PDTL

    # Output image
    # print(im0_temp1.max())
    # print(im0_temp1.min())

    # im0_temp1 = im0_temp1 - np.min(im0_temp1[np.nonzero(im0_temp1)])
    # im0_temp1 = (im0_temp1-np.min(im0_temp1))/(im0_temp1.max()-np.min(im0_temp1))

    im0_temp1 *= (255 / im0_temp1.max())  # 0-1 is spread to 0-255

    im0_temp1 = (im0_temp1).astype(np.uint8)

    # cv2.imshow("img", im0_temp1)
    # cv2.waitKey()

    return im0_temp1

############################################
def pdtl_tile(tile_inp):

    [height, width, dim] = tile_inp.shape

    tile_size = height
    TILE_SQR = tile_size*tile_size

    # Reshape into [dim,tile_size*tile_size]
    tile_reshp = np.transpose(np.reshape(tile_inp, (TILE_SQR, dim)))  # Size: dim x TILE_SQR

    #Reference vector
    ref_vect_0 = np.mat(np.mean(tile_reshp, axis=1))  # Size: 1xdim

    #### Spectral distance ####
    tile_euc_ref = tile_reshp.copy()
    tile_euc_ref[:,:] = np.transpose(ref_vect_0)
    tile_euc = np.linalg.norm(tile_reshp-tile_euc_ref, axis=0)
    tile_euc_sort = np.abs(np.sort(tile_euc))
    ### ------------------ ###

    #### Extract non-zero distances for pdtl ####
    zero_loc = np.where(tile_euc_sort <= 0.0)
    if len(zero_loc)>1:
        tile_euc_sort = tile_euc_sort[zero_loc:TILE_SQR]
        n_elem = tile_euc_sort.shape[0]
    else:
        n_elem = TILE_SQR
    ### ------------------ ###

    #### Log space - x and y coordinates of pdtl####
    counts = np.log10(np.arange(n_elem, dtype=np.float64)+1)
    zero_loc = np.where(tile_euc_sort == 0.0)
    # tile_euc_sort[zero_loc] = 1
    tile_euc_log = np.log10(tile_euc_sort)
    ### ------------------ ###

    #### T ####
    thresh_tail = 0.01
    counts_tail = counts[n_elem-10]
    err_cnt_tail = np.square(counts_tail - counts)
    where_tail = np.min(np.where(err_cnt_tail < thresh_tail))
    PDTL = tile_euc_log[n_elem-1] - tile_euc_log[where_tail]

    return PDTL

########################################################################################################################
def complexity_cd(img_multi_0, img_multi_1, Tile_size):

    ## Image paramters
    [height, width, dim] = img_multi_0.shape
    tile_size = Tile_size
    TILE_SQR = tile_size*tile_size

    ## Output allocation
    img_out = np.zeros((height, width), dtype=np.float64)

    ## Processing tile-by-tile
    for i in range(0, height, tile_size):

        if (i + tile_size) > height:
            break

        for j in range(0, width, tile_size):

            if (j + tile_size) > width:
                break

            # H = height, W = width, T = H*W, d = dimension/bands #

            # ---------------- Select tiles from Image 0 and Image 1
            im0_tile = img_multi_0[i:i + tile_size, j:j + tile_size, :]  # HxWxd
            im1_tile = img_multi_1[i:i + tile_size, j:j + tile_size, :]  # HxWxd

            # Reshape into [dim,tile_size*tile_size]
            im0_tile_reshp = np.transpose(np.reshape(im0_tile, (TILE_SQR, dim)))  # dxT
            im1_tile_reshp =np.transpose(np.reshape(im1_tile, (TILE_SQR, dim)))   # dxT

            ############################################################################################################
            ######################################--- FINDING ENDMEMBERS ---############################################
            # First two endmembers -- shortest and longest vector
            dist_vect_0 = np.sum(np.square(im0_tile_reshp), axis=0)
            dist_vect_1 = np.sum(np.square(im1_tile_reshp), axis=0)

            endmem_pos_0 = np.zeros(dim)
            endmem_pos_0[0] = np.argmax(dist_vect_0)  # longest
            endmem_pos_0[1] = np.argmin(dist_vect_0)   # shortest
            endmem_pos_1 = np.zeros(dim)
            endmem_pos_1[0] = np.argmax(dist_vect_1)  # longest
            endmem_pos_1[1] = np.argmin(dist_vect_1)  # shortest

            Id_mat = np.identity(dim)
            k_endm = 0

            ### Find remaining endmembers
            for i_endmemb_calc in range(2,len(endmem_pos_0)):
                # TILE 1
                diff_endm_0 = np.transpose(np.mat(im0_tile_reshp[:,endmem_pos_0[k_endm]] - im0_tile_reshp[:,endmem_pos_0[k_endm+1]]))  # Size: [dim x 1]
                proj_vect_0 = 1/(np.matmul(np.transpose(diff_endm_0),diff_endm_0))
                proj_vect_0 = np.matmul(proj_vect_0,np.transpose(diff_endm_0))
                proj_vect_0 = Id_mat - np.matmul(diff_endm_0,proj_vect_0)
                proj_img_0 = np.matmul(np.mat(proj_vect_0), np.mat(im0_tile_reshp))

                # TILE 2
                diff_endm_1 = np.transpose(np.mat(im1_tile_reshp[:,endmem_pos_1[k_endm]] - im1_tile_reshp[:,endmem_pos_1[k_endm+1]]))  # Size: [dim x 1]
                proj_vect_1 = 1/(np.matmul(np.transpose(diff_endm_1), diff_endm_1))  # Size: [dim x 1]
                proj_vect_1 = np.matmul(proj_vect_1, np.transpose(diff_endm_1))  # Size: [dim x 1]
                proj_vect_1 = Id_mat - np.matmul(diff_endm_1, proj_vect_1)  # Size: [dim x dim]
                proj_img_1 = np.matmul(proj_vect_1, im1_tile_reshp)  # dim x TILE_SQR

                endmem_pos_0[k_endm+2] = np.argmax(np.sum(np.square(np.transpose(proj_img_0) - np.transpose(proj_img_0[:,endmem_pos_0[k_endm+1]])), axis=0))  # TILE_SQR x dim
                endmem_pos_1[k_endm+2] = np.argmax(np.sum(np.square(np.transpose(proj_img_1) - np.transpose(proj_img_1[:,endmem_pos_1[k_endm+1]])), axis=0))  # TILE_SQR x dim

                k_endm = k_endm + 1
            #########################################-------------------################################################
            ############################################################################################################
            num_endmem = len(endmem_pos_0)
            print(endmem_pos_0)
            print(endmem_pos_1)
            endmem_0 = im0_tile_reshp[:,[endmem_pos_0]]
            endmem_0 = np.reshape(endmem_0, (dim, len(endmem_pos_0)))
            endmem_1 = im1_tile_reshp[:, [endmem_pos_1]]
            endmem_1 = np.reshape(endmem_1, (dim, len(endmem_pos_1)))

            # Local gram function
            gram_fn_loc_0 = np.zeros(num_endmem+1, dtype=np.float64)
            gram_fn_loc_0[0:1] = 0
            gram_fn_loc_1 = gram_fn_loc_0.copy()

            for i_em in range(2,num_endmem):
                arr_endmem_0 = np.zeros((dim,i_em))
                arr_endmem_1 = arr_endmem_0.copy()
                arr_endmem_0 = endmem_0[:,0:i_em]
                arr_endmem_1 = endmem_1[:,0:i_em]

                if i_em>2:
                    loc_gram_0 = local_gram(np.transpose(arr_endmem_0))
                    loc_gram_1 = local_gram(np.transpose(arr_endmem_1))
                    gram_fn_loc_0[i_em] = np.sqrt(abs(np.linalg.det(loc_gram_0)))
                    gram_fn_loc_1[i_em] = np.sqrt(abs(np.linalg.det(loc_gram_1)))

            tile_metrics_0 = Volume_metrics(gram_fn_loc_0)
            tile_metrics_1 = Volume_metrics(gram_fn_loc_1)
            diff_cmplx = abs(tile_metrics_0[2] - tile_metrics_0[2])

            img_out[i:i + tile_size, j:j + tile_size] = diff_cmplx

    # Output image
    img_out *= (255 / img_out.max())  # 0-1 is spread to 0-255
    img_out = np.array(img_out).astype(np.uint8)

    cv2.imshow('img_out', img_out)
    cv2.waitKey()

    return img_out
#############################################################
def local_gram(endmem):

    [dim, npts] = endmem.shape  # dim x npts
    mean_endm = np.mean(endmem, axis=1)  # Size: 1xdim
    euc_endm = np.zeros(npts)
    for i in range(0,npts):
        euc_endm[i] = np.linalg.norm(np.transpose(mean_endm) - endmem[:,i])  # Norm - Euclidean distance
    mindist_pos = np.argmin(euc_endm)  # Min_dist location
    mindist = euc_endm[mindist_pos] # Minimum distance

    otherpix = np.zeros((dim, npts-1), dtype = np.float64)
    index_pix = np.ones(npts)
    index_pix[mindist_pos] = 0
    rows_pix = np.where(index_pix=1)
    nearpix = endmem[:,rows_pix]

    G = gram_matrix(mean_endm, nearpix)

    return G

############################################################
def gram_matrix(mean_endm, nearpix):

    n_bands = len(mean_endm)
    neigh_bands = nearpix.shape[0]
    neigh_size = nearpix.shape[1]

    diff_vec = np.zeros(n_bands, dtype=np.float64)
    diff_vec_j = np.zeros(neigh_bands, dtype=np.float64)
    Gram = np.zeros((neigh_size,neigh_size), dtype=np.float64)

    for i in range(0, neigh_size-1):
        diff_vec = nearpix[:,i] - mean_endm
        Gram[i,i] = np.matmul(np.transpose(diff_vec), diff_vec)

        for j in range(i+1, neigh_size):
            diff_vec_j = nearpix[:,j] - mean_endm
            Gram[i,j] = np.matmul(np.transpose(diff_vec), diff_vec_j)
            Gram[j,i] = Gram[i,j]

    diff_vec = nearpix[:, neigh_size] - mean_endm
    Gram[neigh_size,neigh_size] = np.matmul(np.transpose(diff_vec), diff_vec)

    return Gram

############################################################
def Volume_metrics(tile_vol):

    metrics_count = 3
    metrics = np.zeros(metrics_count, dtype=np.float64)
    N = len(tile_vol)

    tot_vol = np.sum(tile_vol)
    perc_info = np.zeros(N, dtype=np.float64)
    perc_info[0] = tile_vol[0]/tot_vol

    thresh = 0.90
    peak = np.max(tile_vol)

    for i in range(1, N):
        perc_info[i] = perc_info[i-1] + (tile_vol[i]/tot_vol)

    vol_index = np.where(perc_info>thresh)
    areauc = tot_vol

    metrics[0] = vol_index[0]
    metrics[1] = areauc
    metrics[2] = peak

    return metrics

########################################################################################################################
def outliershift_cd(img_multi_0, img_multi_1, Tile_size):

    [height, width, dim] = img_multi_0.shape

    # Check if images are equal in size
    [height1, width1, dim1] = img_multi_1.shape
    if (height != height1) or (width != width1) or (dim != dim1):
        raise Exception("Images not equal in size")

    tile_size = Tile_size
    TILE_SQR = tile_size*tile_size

    im0_temp1 = np.zeros((height, width), dtype=np.float64)
    h1 = height/100  # For status percentage

    # ---------------- Processing tile-by-tile
    for i in range(0, height, tile_size):

        # Show progress percentage
        sys.stdout.write('\r')
        sys.stdout.write("Status:    %d%%" % ((i + 1) / h1))
        sys.stdout.flush()

        if (i + tile_size) > height:
            break

        for j in range(0, width, tile_size):

            if (j + tile_size) > width:
                break


            # ---------------- Select tiles from Image 0 and Image 1
            im0_tile = img_multi_0[i:i + tile_size, j:j + tile_size, :]
            im1_tile = img_multi_1[i:i + tile_size, j:j + tile_size, :]


            # Reshape into [tile_size*tile_size,dim]
            im0_tile_reshp = np.reshape(im0_tile, (TILE_SQR,dim))
            im1_tile_reshp = np.reshape(im1_tile, (TILE_SQR,dim))

            # Mean-vector
            t1 = time.time()
            mean_vect_x = np.mean(im0_tile_reshp,0)
            mean_vect_y = np.mean(im0_tile_reshp,0)
            outlier_dist_x = np.zeros(TILE_SQR,dtype=np.float64)
            outlier_dist_y = np.zeros(TILE_SQR,dtype=np.float64)

            for i_tile in range(0,TILE_SQR):
                outlier_dist_x[i_tile] = np.sqrt(np.sum(np.square(np.mat(mean_vect_x)-np.mat(im0_tile_reshp[i_tile,:]))))
                outlier_dist_y[i_tile] = np.sqrt(np.sum(np.square(np.mat(mean_vect_y)-np.mat(im1_tile_reshp[i_tile,:]))))

            diff_outlier_vect = abs(outlier_dist_x.max() - outlier_dist_y.max())
            # print("Time for finding 2 means is %.2f" % (time.time() - t1))

            im0_temp1[i:i + tile_size, j:j + tile_size] = diff_outlier_vect

    # Output image
    im0_temp1 *= (255 / im0_temp1.max())  # 0-1 is spread to 0-255
    img_out = np.array(im0_temp1).astype(np.uint8)

    return img_out
########################################################################################################################
def meanshift_cd(img_multi_0, img_multi_1, Tile_size):

    [height, width, dim] = img_multi_0.shape

    # Check if images are equal in size
    [height1, width1, dim1] = img_multi_1.shape
    if (height != height1) or (width != width1) or (dim != dim1):
        raise Exception("Images not equal in size")

    tile_size = Tile_size

    im0_temp1 = np.zeros((height, width), dtype=np.float64)
    h1 = height/100  # For status percentage

    # ---------------- Processing tile-by-tile
    for i in range(0, height, tile_size):

        # Show progress percentage
        sys.stdout.write('\r')
        sys.stdout.write("Status:    %d%%" % ((i + 1) / h1))
        sys.stdout.flush()

        if (i + tile_size) > height:
            break

        for j in range(0, width, tile_size):

            if (j + tile_size) > width:
                break


            # ---------------- Select tiles from Image 0 and Image 1
            im0_tile = img_multi_0[i:i + tile_size, j:j + tile_size, :]
            im1_tile = img_multi_1[i:i + tile_size, j:j + tile_size, :]

            # Mean-vector
            t1 = time.time()
            mean_vect_x = []
            mean_vect_y = []
            for i_dim in range(0, dim):
                mean_vect_x.append(im0_tile[:, :, i_dim].mean())
                mean_vect_y.append(im1_tile[:, :, i_dim].mean())
            diff_mean_vect = np.sqrt(np.sum(np.square(np.mat(mean_vect_x)-np.mat(mean_vect_y))))
            # print("Time for finding 2 means is %.2f" % (time.time() - t1))

            im0_temp1[i:i + tile_size, j:j + tile_size] = diff_mean_vect

    # Output image
    im0_temp1 *= (255 / im0_temp1.max())  # 0-1 is spread to 0-255
    img_out = np.array(im0_temp1).astype(np.uint8)

    return img_out
########################################################################################################################
def chronochrome_cd(img_multi_0, img_multi_1, alg):

    [height, width, dim] = img_multi_0.shape

    # Check if images are equal in size
    [height1, width1, dim1] = img_multi_1.shape
    if (height != height1) or (width != width1) or (dim != dim1):
        raise Exception("Images not equal in size")

    # Mean-vector
    t1 = time.time()
    mean_vect_x = []
    mean_vect_y = []
    for i in range(0, dim):
        mean_vect_x.append(img_multi_0[:, :, i].mean())
        mean_vect_y.append(img_multi_1[:, :, i].mean())
    mean_vect_x = np.mat(np.array(mean_vect_x))
    mean_vect_x = mean_vect_x.T  # Size: dim x 1
    mean_vect_y = np.mat(np.array(mean_vect_y))
    mean_vect_y = mean_vect_y.T  # Size: dim x 1
    print("Time for finding 2 means is %.2f" % (time.time() - t1))

    # Covariance
    t1 = time.time()
    C1 = cov_images(img_multi_0)
    print("Time for C1 is %.2f" % (time.time() - t1))
    t1 = time.time()

    if len(alg) == 0:
        raise Exception("missing parameter - algorithm")

    if alg == 'CC':
        C12 = cov_images(img_multi_0, img_multi_1)
    elif alg == 'CE':
        C2 = cov_images(img_multi_1)
    else:
        raise Exception("Use 'CC' or 'CE' as the algorithm parameter")
    print("Time for C12 is %.2f" % (time.time() - t1))

    # Global predictor
    t1 = time.time()
    if alg == 'CC':
        A12 = np.dot(C12, np.linalg.inv(C1))
    else:
        A12 = np.dot(np.sqrt(C2), np.linalg.inv(np.sqrt(C1)))
    b = mean_vect_y - np.dot(A12, mean_vect_x)
    print("Time for predictor b is %.2f" % (time.time() - t1))

    # Output image
    t1 = time.time()
    img_pred = np.zeros((height, width, dim),dtype=np.float64)
    img_residual = img_pred.copy()
    img_out_2 = np.zeros((height, width))
    img_out_3 = np.zeros((height, width))

    h1 = height/100  # For status percentage
    for i in range(0, height):

        # Show progress percentage
        sys.stdout.write('\r')
        sys.stdout.write("Status:    %d%%" % ((i+1)/h1))
        sys.stdout.flush()

        for j in range(0, width):
            pred = (b + np.dot(A12, (np.mat(img_multi_0[i,j,:])).T)).T
            change_res = abs(pred - np.mat(img_multi_1[i,j,:]))
            max_val = np.max(change_res)
            img_out_3[i,j] = max_val

    img_out_3 = img_out_3.astype(np.uint8)

    print("\n Time for propogating is %.2f" % (time.time() - t1))

    return img_out_3
########################################################################################################################
def image_diff(img_multi_0, img_multi_1):

    [height, width, dim] = img_multi_0.shape  # Image-size

    img_out = np.zeros((height,width,dim),dtype=np.int16)  # Output definition

    # Type-cast to include negatives
    img_multi_0 = img_multi_0.astype(np.int16)
    img_multi_1 = img_multi_1.astype(np.int16)

    # Differencing per band
    for i in range(0,dim):
        img_out[:,:,i] =  np.abs(np.subtract(img_multi_0[:,:,i],img_multi_1[:,:,i]))

    # Maximum from all dimensions for each spatial pixel location
    max_img = np.zeros((dim,height*width),dtype=np.int16)
    for i in range(0,dim):
        max_img[i,:] = np.ravel(img_out[:,:,i])
    img_temp = np.amax(max_img,axis=0)  # Maximum of four bands
    img_final = (img_temp.reshape((height,width))).astype(np.uint8)  #Reshaping into (height,width) array

    # cv2.namedWindow('img_final', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('img_final',700,700)
    # cv2.imshow('img_final', img_final)
    # cv2.waitKey()

    return img_final

########################################################################################################################
def image_ratio(img_multi_0, img_multi_1):

    #Datatype conversion
    [height, width, dim] = img_multi_0.shape
    img_out = np.zeros((height, width, dim), dtype=np.uint8)
    img_multi_0 = img_multi_0.astype(np.float64)
    img_multi_1 = img_multi_1.astype(np.float64)

    for i in range(0, dim):
        img_out_temp = np.zeros((2,height*width), dtype=np.float64)
        img_out_temp[0,:] = np.ravel(img_multi_0[:, :, i])  # ith band of 1st image as first row
        img_out_temp[1,:] = np.ravel(img_multi_1[:, :, i])  # ith band of 2nd image as second row
        img_out_temp = np.sort(img_out_temp,axis=0)  # Sorting along columns(location invariant)
        img_out_temp_1 = np.zeros((1,height*width), dtype=np.float64)
        b_zeros = np.where(img_out_temp[1, :] ==0)
        (img_out_temp[1, :])[b_zeros] = 1
        img_out_temp_1[0,:] = np.divide(img_out_temp[0,:], img_out_temp[1,:])  # Ratio pixel by pixel
        img_out_temp_1 *= (255/img_out_temp_1.max())  # 0-1 is spread to 0-255
        img_out[:,:,i] = (img_out_temp_1.reshape((height,width)))  # converting to uint8


    max_img = np.zeros((dim, height * width), dtype=np.uint8)
    for i in range(0, dim):
        max_img[i, :] = np.ravel(img_out[:, :, i])
    img_temp = np.amin(max_img, axis=0)
    img_final = (255 - (img_temp.reshape((height, width)))).astype(np.uint8)
    return img_final
########################################################################################################################
def graph_theoretic(img_multi_0, img_multi_1,*args,Tile_size,K_NN):

    print("*************** Graph-theoretic Approach ***************")
    # TILE-images PARAMETERS
    tile_size = Tile_size  # Tile-size in pixels
    # K_NN is the 'k'-nearest neighbors

    # Input-image & Adjacency matrix dimension
    [height, width, dim] = img_multi_0.shape
    height_adj_mat = tile_size * tile_size  # Adjacency matrix height
    width_adj_mat = tile_size * tile_size  # Adjacency matrix width

    #---------------- Process whole image OR part of the image, based on input arguments
    if len(args)>0:
        xy_startend = args[0]
        startx = xy_startend[0]
        starty = xy_startend[1]
        endx = xy_startend[2]
        endy = xy_startend[3]
    else: # default case
        startx = 0
        starty = 0
        endx = height
        endy = width

    #---------------- Output image
    im0_temp1 = np.zeros((endx-startx, endy-starty), dtype=np.float64)
    im1_temp1 = im0_temp1.copy()
    im0_temp2 = im0_temp1.copy()
    im1_temp2 = im0_temp2.copy()
    i_temp = 0  # for assigning NEV values to im1_temp1 and 2 arrays.
    j_temp = 0

    tile_no = 0
    tile_no_break = -1  ## Default = -1: For processing a user-def number of tiles
    print("Two 'change-images' -- Height,Width,Tile_size: (%d,%d),%d" % (height, width, tile_size))
    print("Start processing...")
    start_time = time.time()

    total_vertices = K_NN * height_adj_mat

    # ---------------- Processing tile-by-tile
    for i in range(startx, endx, tile_size):

        if (i + tile_size) > height:
            break

        for j in range(starty, endy, tile_size):

            if (j + tile_size) > width:
                break
            tile_no = tile_no + 1
            if tile_no == tile_no_break:
                break


            # ---------------- Select tiles from Image 0 and Image 1
            im0_tile = img_multi_0[i:i + tile_size, j:j + tile_size,:]
            im1_tile = img_multi_1[i:i + tile_size, j:j + tile_size,:]

            # ---------------- Unwind the image-tile into a MN x dim vector (MN = M*N = number of pixels)
            im0tile_vector = np.zeros((dim,tile_size*tile_size))
            im1tile_vector = im0tile_vector.copy()
            for i_band in range(0,dim):
                im0tile_vector[i_band,:] = np.ravel(im0_tile[:,:,i_band])
                im1tile_vector[i_band,:] = np.ravel(im1_tile[:,:,i_band])

            #---------------- Weighted Ajdacency matrix for each image tile
            Weight_adj_matrix_0 = np.zeros((height_adj_mat, width_adj_mat), dtype=np.float64)  # for im0_tile
            Weight_adj_matrix_1 = np.zeros((height_adj_mat, width_adj_mat), dtype=np.float64)  # for im1_tile

            ################################## Alternative/faster way for Adjancecy matrix
            Weight_adj_matrix_0_opt = Weight_adj_matrix_0.copy()
            Weight_adj_matrix_1_opt = Weight_adj_matrix_1.copy()

            wadj_mat_0 = np.zeros((height_adj_mat, width_adj_mat, dim), dtype=np.float64)
            wadj_mat_1 = wadj_mat_0.copy()

            for i_wg in range(0,height_adj_mat):
                for i_wg_band in range(0, dim):
                    wadj_mat_0[:, i_wg, i_wg_band] = im0tile_vector[i_wg_band, :] - im0tile_vector[i_wg_band, i_wg]
                    wadj_mat_1[:, i_wg, i_wg_band] = im1tile_vector[i_wg_band, :] - im1tile_vector[i_wg_band, i_wg]

            for i_wg_band in range(0, dim):
                Weight_adj_matrix_0_opt = Weight_adj_matrix_0_opt + np.square(wadj_mat_0[:,:,i_wg_band])
                Weight_adj_matrix_1_opt = Weight_adj_matrix_1_opt + np.square(wadj_mat_1[:, :, i_wg_band])

            Weight_adj_matrix_0_opt = np.sqrt(Weight_adj_matrix_0_opt)
            Weight_adj_matrix_1_opt = np.sqrt(Weight_adj_matrix_1_opt)

            for i_wg in range(0, height_adj_mat):

                Weight_adj_matrix_0_opt[i_wg,:] = np.sort(Weight_adj_matrix_0_opt[i_wg,:])
                Weight_adj_matrix_0_opt[i_wg,(K_NN+1):] = 0
                Weight_adj_matrix_1_opt[i_wg, :] = np.sort(Weight_adj_matrix_1_opt[i_wg, :])
                Weight_adj_matrix_1_opt[i_wg, (K_NN + 1):] = 0

            ##---------------- Calculate NEV - Normalized Edge Volume
            NEV0_opt = np.sum(Weight_adj_matrix_0_opt) / (np.count_nonzero(Weight_adj_matrix_0_opt))
            NEV1_opt = np.sum(Weight_adj_matrix_1_opt) / (np.count_nonzero(Weight_adj_matrix_1_opt))

            if (np.count_nonzero(Weight_adj_matrix_0_opt))==0:
                NEV0_opt = 0
            if (np.count_nonzero(Weight_adj_matrix_1_opt))==0:
                NEV1_opt = 0

            # ---------------- Calculate SDEL - Standard Deviation of Edge Lengths
            SDEL0_opt = np.std(Weight_adj_matrix_0_opt)
            SDEL1_opt = np.std(Weight_adj_matrix_1_opt)

            # ---------------- Print status after each TILE is processed
            print(
                'processing-OPTIMIZED tile no. Row/Column: (%d of %d)/(%d of %d)     NEV0_opt = %.2f   NEV1_opt = %.2f   SDEL0_opt = %.2f  SDEL1_opt =  = %.2f' % (
                    i, endx, j, endy, NEV0_opt, NEV1_opt, SDEL0_opt, SDEL1_opt))

            im0_temp1[i_temp:i_temp + tile_size, j_temp:j_temp + tile_size] = abs(NEV0_opt - NEV1_opt)  # positive float value
            im1_temp1[i_temp:i_temp + tile_size, j_temp:j_temp + tile_size] = abs(SDEL0_opt - SDEL1_opt)  # positive float value
            j_temp += tile_size

        i_temp += tile_size
        j_temp = 0

    total_time = time.time() - start_time
    print("Processed two 'change-images': Height,Width = (%d,%d)  Tile_size = %d   K_nn = %d     Total time taken = %.2f" % (
    height, width, tile_size, K_NN,total_time))

    aa0 = im0_temp1.max()
    print(aa0)
    aa1 = np.argmax(im0_temp1)
    print(aa1)
    aa2 = im1_temp1.max()
    im0_temp1 *= (255 / im0_temp1.max())  # 0-1 is spread to 0-255
    im0_temp1 = np.array(im0_temp1).astype(np.uint8)

    return im0_temp1


def cd_pca(img_multi_0, img_multi_1):

    # PRINCICPAL COMPONENT ANALYSIS

    [height, width, dim] = img_multi_0.shape

    img_multi = np.zeros((height,width,dim*2))
    img_multi[:, :, 0:dim] = img_multi_0
    img_multi[:, :, dim:(dim*2)] = img_multi_1

    [height, width, dim] = img_multi.shape

    # Mean-vector
    t1 = time.time()
    mean_vect = []
    for i in range(0, dim):
        mean_vect.append(img_multi[:, :, i].mean())
    mean_vect = np.mat(np.array(mean_vect))
    mean_vect = mean_vect.T  # Size: dim x 1
    print("Time for finding 2 means is %.2f" % (time.time() - t1))

    # Covariance
    t1 = time.time()
    C1 = cov_images(img_multi)
    print("Time for C1 is %.2f" % (time.time() - t1))

    t1 = time.time()
    # EIGEN-VALUE DECOMPOSITION
    eig_val, eig_vec = np.linalg.eig(C1)
    print("Time for Eigen decomposition is %.2f" % (time.time() - t1))
    eig_val_sorted_ind = sorted(range(len(eig_val)), key=lambda k: eig_val[k])
    eig_vec_sorted = eig_vec.copy()
    for i_sort in range(0,len(eig_val_sorted_ind)):
        ind_sort = eig_val_sorted_ind[-(i_sort+1)]
        eig_vec_sorted[:,i_sort] = eig_vec[:,ind_sort]
    eig_vec_sorted = np.mat(eig_vec_sorted)

    PC_bands = np.zeros((height,width,dim))
    for i in range(0,height):
        for j in range(0,width):
            PC_bands[i,j,:] = np.dot(np.mat(img_multi[i,j,:]),eig_vec_sorted)

    PC_bands = (np.array(PC_bands)).astype(np.uint8)


    return PC_bands[:,:,7]


def cov_images(img1,*args):
    # Covariance
    [height, width, dim] = img1.shape
    C = np.zeros((dim, dim))
    if len(args)>0:
        img2 = img1.copy()
        img2[:,:,:] = args[0]
        # Mean-centering
        for i in range(0, dim):
            img1[:,:,i] = img1[:,:,i] - img1[:,:,i].mean()
            img2[:,:,i] = img2[:,:,i] - img2[:,:,i].mean()
        # Cross-Covariance
        for i in range(0, dim):
            for j in range(0, dim):
                C[i, j] = np.sum(np.multiply(img1[:, :, i], img2[:, :, j]))
    else:
        # Mean-centering
        for i in range(0, dim):
            img1[:,:,i] = img1[:,:,i] - img1[:,:,i].mean()

        for i in range(0,dim):
            for j in range(0,dim):
                C[i,j] =np.sum(np.multiply(img1[:,:,i],img1[:,:,j]))
    C *= 1. / np.float64(height * width)
    return C
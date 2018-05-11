#GG
#Author: Jobin J Mathew
#All copyrights reserved to the author and DIRS group at Chester F. Carlson Center for Imaging Science (CIS), at Rochester Institute
#Technology (RIT), Rochester, NY.

#A Change Detection (CD) GUI (Graphical User Interface) for the visual and qualitative evaluation of
#various change detection algorithms applied on single-band, multi-spectral, hyper-spectral images.

#********************************************************************************************************************#
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import JJMCIS_Algorithms as cdfn
import JJMCIS_Functions
import time
import JJMCIS_GTEvaluation as GTEval
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import os


root = Tk()

#### Variable definitions ###
file_name_1 = StringVar()
file_name_2 = StringVar()
file_name_3 = StringVar()
ADD_NOISE = False
NOISE_VAL = 0

####### Change Detection Algorithm  #########
def CD_algorithm():

    ## Print user-selected parameters
    print("NOISE_VAL is %d" % (NOISE_VAL))
    print("TILE_SIZE is %d" % (TILE_SIZE))
    print("TILE_or_SLIDE is %d" % (TILE_or_SLIDE))  # 1 - Tile, 0 - Slide
    print("ALG_CHECK is ",end="")
    print(", ".join([str(a) for a in ALG_CHECK]))

    ###########--- READING INPUT IMAGES browsed by user from the toolbox ################
    img_multi_0 = JJMCIS_Functions.read_envi_img(file_name_1)
    img_multi_1 = JJMCIS_Functions.read_envi_img(file_name_2)
    img_p0 = img_multi_0[:, :, 3]

    (height, width, dim) = img_multi_0.shape


    # Display figure initialization
    my_dpi = 96
    fig = plt.figure(figsize=(200/my_dpi,200/my_dpi), dpi=my_dpi)
    plt.ion()
    canvas = FigureCanvasTkAgg(fig,master=root)
    plot_roc_update = canvas.get_tk_widget().place(x=480,y=350)
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.tick_params(axis='both', which='major', labelsize=7)


    ###############--------Algorithm indices--------######
    # #indices [0-ALL algorithm, 1-PDTL, 2-Complexity, 3-Mean-shift, 4-Outlier-Shift, 5-Graph-based, 6-Chronochrome,
    # 7-Covariance Equalization, 8-PCA, 9-Image Diff & Image Ratioing

    # -------------------## RUNNING ALGORITHMS ##-------------------------#
    # --------------------------------------------------------------------#
    spc = ' '
    Tile_size = TILE_SIZE

    ################******** PDTL Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[1]==1:
        print('PDTL....')
        xy_startend = [0, 0, height, width]
        img_out_1 = cdfn.pdtl_cd(img_multi_0, img_multi_1, Tile_size=Tile_size)
        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out = Image.fromarray(img_out_1)
        img_alg_out = img_alg_out.resize((200, 200), Image.ANTIALIAS)
        img_alg_out = ImageTk.PhotoImage(img_alg_out)
        Img1Out = Label(image=img_alg_out)
        Img1Out.image = img_alg_out
        Img1Out.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        [fa_list, hit_list, auroc] = GTEval.gt_eval(img_out_1, file_name_3, Tile_size, xy_startend)

        lab_auc_1 = Label(text="PDTL " + spc * 46 + "%.2f" % (abs(auroc)), relief=RIDGE)
        lab_auc_1.place(x=715, y=355)

        plt.plot(fa_list, hit_list, label='PDTL', hold=True)
    else:
        lab_auc_1 = Label(text="PDTL " + spc * 46 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_1.place(x=715, y=355)
    # --------------------------------------------------------------------#

    ################******** Complexity Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[2]==1:
        print('Complexity....')
        xy_startend = [0, 0, height, width]
        img_out_2 = cdfn.complexity_cd(img_multi_0, img_multi_1, Tile_size=Tile_size)
        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out = Image.fromarray(img_out_2)
        img_alg_out = img_alg_out.resize((200, 200), Image.ANTIALIAS)
        img_alg_out = ImageTk.PhotoImage(img_alg_out)
        Img1Out = Label(image=img_alg_out)
        Img1Out.image = img_alg_out
        Img1Out.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out = np.dstack((img_out_2, img_out_2, img_p0))
        img_mrg_out = Image.fromarray(img_mrg_out)
        img_mrg_out = img_mrg_out.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out = ImageTk.PhotoImage(img_mrg_out)
        Img1OutMerge = Label(image=img_mrg_out)
        Img1OutMerge.image = img_mrg_out
        Img1OutMerge.place(x=250, y=350)
        ##########################################################################################
        [fa_list, hit_list, auroc] = GTEval.gt_eval(img_out_2, file_name_3, Tile_size, xy_startend)

        lab_auc_2 = Label(text="CMPX" + spc * 45 + "%.2f" % (abs(auroc)), relief=RIDGE)
        lab_auc_2.place(x=715, y=373)

        plt.plot(fa_list, hit_list, label='Complexity', hold=True)
    else:
        lab_auc_2 = Label(text="CMPX" + spc * 45 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_2.place(x=715, y=373)
    # --------------------------------------------------------------------#

    ################******** Mean-shift Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[3]==1:
        print('Mean....')
        xy_startend = [0, 0, height, width]
        img_out_3 = cdfn.meanshift_cd(img_multi_0, img_multi_1, Tile_size=Tile_size)

        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out_3 = Image.fromarray(img_out_3)
        img_alg_out_3 = img_alg_out_3.resize((200, 200), Image.ANTIALIAS)
        img_alg_out_3 = ImageTk.PhotoImage(img_alg_out_3)
        Img1Out_3 = Label(image=img_alg_out_3)
        Img1Out_3.image = img_alg_out_3
        Img1Out_3.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out_3 = np.dstack((img_out_3, img_out_3, img_p0))
        img_mrg_out_3 = Image.fromarray(img_mrg_out_3)
        img_mrg_out_3 = img_mrg_out_3.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out_3 = ImageTk.PhotoImage(img_mrg_out_3)
        Img1OutMerge_3 = Label(image=img_mrg_out_3)
        Img1OutMerge_3.image = img_mrg_out_3
        Img1OutMerge_3.place(x=250, y=350)
        ##########################################################################################

        [fa_list_3, hit_list_3, auroc_3] = GTEval.gt_eval(img_out_3, file_name_3, Tile_size, xy_startend)

        lab_auc_3 = Label(text="MEAN " + spc * 44 + "%.2f" % (abs(auroc_3)), relief=RIDGE)
        lab_auc_3.place(x=715, y=391)

        plt.plot(fa_list_3, hit_list_3, label='Mean', hold=True)
    else:
        lab_auc_3 = Label(text="MEAN " + spc * 44 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_3.place(x=715, y=391)
    # --------------------------------------------------------------------#

    ################******** Outlier-shift Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[4]==1:
        print('Outlier....')
        xy_startend = [0, 0, height, width]
        img_out_4 = cdfn.outliershift_cd(img_multi_0, img_multi_1, Tile_size=Tile_size)

        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out_4 = Image.fromarray(img_out_4)
        img_alg_out_4 = img_alg_out_4.resize((200, 200), Image.ANTIALIAS)
        img_alg_out_4 = ImageTk.PhotoImage(img_alg_out_4)
        Img1Out_4 = Label(image=img_alg_out_4)
        Img1Out_4.image = img_alg_out_4
        Img1Out_4.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out_4 = np.dstack((img_out_4, img_out_4, img_p0))
        img_mrg_out_4 = Image.fromarray(img_mrg_out_4)
        img_mrg_out_4 = img_mrg_out_4.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out_4 = ImageTk.PhotoImage(img_mrg_out_4)
        Img1OutMerge_4 = Label(image=img_mrg_out_4)
        Img1OutMerge_4.image = img_mrg_out_4
        Img1OutMerge_4.place(x=250, y=350)
        ##########################################################################################
        [fa_list_4, hit_list_4, auroc_4] = GTEval.gt_eval(img_out_4, file_name_3, Tile_size, xy_startend)

        lab_auc_4 = Label(text="OUTL" + spc * 46 + "%.2f" % (abs(auroc_4)), relief=RIDGE)
        lab_auc_4.place(x=715, y=409)

        plt.plot(fa_list_4, hit_list_4, label='Outlier', hold=True)
    else:
        lab_auc_4 = Label(text="OUTL" + spc * 46 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_4.place(x=715, y=409)
    # --------------------------------------------------------------------#

    ################******** Graph-theoretic Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[5]==1:
        print('Graph....')
        t1 = time.time()
        xy_startend = [0,0,height,width]
        K_NN = 30
        img_out = cdfn.graph_theoretic(img_multi_0, img_multi_1,xy_startend,Tile_size=Tile_size,K_NN = K_NN)
        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out = Image.fromarray(img_out)
        img_alg_out = img_alg_out.resize((200, 200), Image.ANTIALIAS)
        img_alg_out = ImageTk.PhotoImage(img_alg_out)
        Img1Out = Label(image=img_alg_out)
        Img1Out.image = img_alg_out
        Img1Out.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out = np.dstack((img_out, img_out, img_p0))
        img_mrg_out = Image.fromarray(img_mrg_out)
        img_mrg_out = img_mrg_out.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out = ImageTk.PhotoImage(img_mrg_out)
        Img1OutMerge = Label(image=img_mrg_out)
        Img1OutMerge.image = img_mrg_out
        Img1OutMerge.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for Graph-theoretic appraoch is %.2f" % (time.time() - t1))
        [fa_list, hit_list, auroc] = GTEval.gt_eval(img_out, file_name_3, Tile_size, xy_startend)

        lab_auc_5 = Label(text="GRPH" + spc * 45 + "%.2f" % (abs(auroc)), relief=RIDGE)
        lab_auc_5.place(x=715, y=427)

        plt.plot(fa_list, hit_list, label='Graph', hold=True, linewidth=2)
        fig.canvas.draw()
        canvas.show()
        plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0., fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=7)
        ##########################################################################################
    else:
        lab_auc_5 = Label(text="GRPH" + spc * 45 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_5.place(x=715, y=427)
    # --------------------------------------------------------------------#

    ################******** Chronochrome Detection Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[6]==1:
        print('CC....')
        t1 = time.time()
        Tile_size = 1  # Since non-tile based algorithm
        xy_startend = [0,0,height,width]
        img_out_6 = cdfn.chronochrome_cd(img_multi_0, img_multi_1,alg='CC')
        # ^^^^^^Image_output_display^^^^^^#
        # img_alg_out_6 = Image.fromarray(img_out_6)
        # img_alg_out_6 = img_alg_out_6.resize((200, 200), Image.ANTIALIAS)
        # img_alg_out_6 = ImageTk.PhotoImage(img_alg_out_6)
        # Img1Out_6 = Label(image=img_alg_out_6)
        # Img1Out_6.image = img_alg_out_6
        # Img1Out_6.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        # img_mrg_out_6 = np.dstack((img_out_6, img_out_6, img_p0))
        # img_mrg_out_6 = Image.fromarray(img_mrg_out_6)
        # img_mrg_out_6 = img_mrg_out_6.resize((200, 200), Image.ANTIALIAS)
        # img_mrg_out_6 = ImageTk.PhotoImage(img_mrg_out_6)
        # Img1OutMerge_6 = Label(image=img_mrg_out_6)
        # Img1OutMerge_6.image = img_mrg_out_6
        # Img1OutMerge_6.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for ChronoChrome detection is %.2f" % (time.time() - t1))
        [fa_list_6, hit_list_6, auroc_6] = GTEval.gt_eval(img_out_6, file_name_3, Tile_size, xy_startend)

        lab_auc_6 = Label(text="CC" + spc * 50 + "%.2f" % (abs(auroc_6)), relief=RIDGE)
        lab_auc_6.place(x=715, y=445)

        plt.plot(fa_list_6, hit_list_6, label='CC', hold=True, linewidth=2)
        fig.canvas.draw()
        canvas.show()
        plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0., fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=7)
        ##########################################################################################
    else:
        lab_auc_6 = Label(text="CC" + spc * 50 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_6.place(x=715, y=445)
    # --------------------------------------------------------------------#

    ################******** Covariance Equalization Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[7]==1:
        print('CE....')
        t1 = time.time()
        Tile_size = 1  # Since non-tile based algorithm
        xy_startend = [0,0,height,width]
        img_out_7 = cdfn.chronochrome_cd(img_multi_0, img_multi_1,alg='CE')
        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out_7 = Image.fromarray(img_out_7)
        img_alg_out_7 = img_alg_out_7.resize((200, 200), Image.ANTIALIAS)
        img_alg_out_7 = ImageTk.PhotoImage(img_alg_out_7)
        Img1Out_7 = Label(image=img_alg_out_7)
        Img1Out_7.image = img_alg_out_7
        Img1Out_7.place(x=20, y=350)
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out_7 = np.dstack((img_out_7, img_out_7, img_p0))
        img_mrg_out_7 = Image.fromarray(img_mrg_out_7)
        img_mrg_out_7 = img_mrg_out_7.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out_7 = ImageTk.PhotoImage(img_mrg_out_7)
        Img1OutMerge_7 = Label(image=img_mrg_out_7)
        Img1OutMerge_7.image = img_mrg_out_7
        Img1OutMerge_7.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for Covariance Equalization is %.2f" % (time.time() - t1))
        [fa_list_7, hit_list_7, auroc_7] = GTEval.gt_eval(img_out_7, file_name_3, Tile_size, xy_startend)

        lab_auc_7 = Label(text="CE" + spc * 51 + "%.2f" % (abs(auroc_7)), relief=RIDGE)
        lab_auc_7.place(x=715, y=463)

        plt.plot(fa_list_7, hit_list_7, label='CE', hold=True, linewidth=2)
        fig.canvas.draw()
        canvas.show()
        plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0., fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=7)
        ##########################################################################################
    else:
        lab_auc_7 = Label(text="CE" + spc * 51 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_7.place(x=715, y=463)
    # --------------------------------------------------------------------#

    ################******** PCA Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[8]==1:
        print('PCA....')
        t1 = time.time()
        Tile_size = 1  # Since non-tile based algorithm
        xy_startend = [0,0,height,width]
        img_out = cdfn.cd_pca(img_multi_0, img_multi_1)
        # ^^^^^^Image_output_display^^^^^^#
        img_alg_out = Image.fromarray(img_out)
        img_alg_out = img_alg_out.resize((200, 200), Image.ANTIALIAS)
        img_alg_out = ImageTk.PhotoImage(img_alg_out)
        Img1Out = Label(image=img_alg_out)
        Img1Out.image = img_alg_out
        Img1Out.place(x=20, y=350)
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out = np.dstack((img_out, img_out, img_p0))
        img_mrg_out = Image.fromarray(img_mrg_out)
        img_mrg_out = img_mrg_out.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out = ImageTk.PhotoImage(img_mrg_out)
        Img1OutMerge = Label(image=img_mrg_out)
        Img1OutMerge.image = img_mrg_out
        Img1OutMerge.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for PCA detection is %.2f" % (time.time() - t1))
        [fa_list, hit_list, auroc] = GTEval.gt_eval(img_out, file_name_3, Tile_size, xy_startend)

        lab_auc_8 = Label(text="PCA" + spc * 48 + "%.2f" % (abs(auroc)), relief=RIDGE)
        lab_auc_8.place(x=715, y=481)

        plt.plot(fa_list, hit_list, label='PCA', hold=True, linewidth=2)
        fig.canvas.draw()
        canvas.show()
        plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0., fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=7)
        ##########################################################################################
    else:
        lab_auc_8 = Label(text="PCA" + spc * 48 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_8.place(x=715, y=481)

    # --------------------------------------------------------------------#

    ################******** Differencing/Ratioing Algorithm
    if ALG_CHECK[0]==1 or ALG_CHECK[9]==1:
        ##########################################################################################
        # ALGORITHM - IMAGE DIFFERENCING
        print('ID & IR ....')
        t1 = time.time()
        Tile_size = 1
        xy_startend = [0,0,height,width]
        img_out_9 = cdfn.image_diff(img_multi_0, img_multi_1)
        ##########################################################################################
        # OUTPUT_DISPLAY
        img_alg_out_9 = Image.fromarray(img_out_9)
        img_alg_out_9 = img_alg_out_9.resize((200, 200), Image.ANTIALIAS)
        img_alg_out_9 = ImageTk.PhotoImage(img_alg_out_9)
        Img1Out_9 = Label(image=img_alg_out_9)
        Img1Out_9.image = img_alg_out_9
        Img1Out_9.place(x=20, y=350)
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out_9 =np.dstack((img_out_9, img_out_9, img_p0))
        img_mrg_out_9 = Image.fromarray(img_mrg_out_9)
        img_mrg_out_9 = img_mrg_out_9.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out_9 = ImageTk.PhotoImage(img_mrg_out_9)
        Img1OutMerge_9 = Label(image=img_mrg_out_9)
        Img1OutMerge_9.image = img_mrg_out_9
        Img1OutMerge_9.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for Image Differencing is %.2f"%(time.time()-t1))
        [fa_list_9, hit_list_9, auroc_9] = GTEval.gt_eval(img_out_9, file_name_3, Tile_size, xy_startend)

        lab_auc_9 = Label(text="IDIF" + spc * 49 + "%.2f" % (abs(auroc_9)), relief=RIDGE)
        lab_auc_9.place(x=715, y=499)

        plt.plot(fa_list_9, hit_list_9, label='Image Diff', hold=True, linewidth=2)
        fig.canvas.draw()
        canvas.show()
        plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0., fontsize=4)
        plt.tick_params(axis='both', which='major', labelsize=7)
        ##########################################################################################
        ##########################################################################################
        # ALGORITHM - IMAGE RATIOING
        t1 = time.time()
        Tile_size = 1
        xy_startend = [0,0,height,width]
        img_out_10 = cdfn.image_ratio(img_multi_0, img_multi_1)
        ##########################################################################################
        # OUTPUT_DISPLAY
        img_alg_out_10 = Image.fromarray(img_out_10)
        img_alg_out_10 = img_alg_out_10.resize((200, 200), Image.ANTIALIAS)
        img_alg_out_10 = ImageTk.PhotoImage(img_alg_out_10)
        Img1Out_10 = Label(image=img_alg_out_10)
        Img1Out_10.image = img_alg_out_10
        Img1Out_10.place(x=20, y=350)
        ##########################################################################################
        # OUTPUT_MERGE_DISPLAY
        img_mrg_out_10 = np.dstack((img_p0, img_out_10, img_p0))
        img_mrg_out_10 = Image.fromarray(img_mrg_out_10)
        img_mrg_out_10 = img_mrg_out_10.resize((200, 200), Image.ANTIALIAS)
        img_mrg_out_10 = ImageTk.PhotoImage(img_mrg_out_10)
        Img1OutMerge_10 = Label(image=img_mrg_out_10)
        Img1OutMerge_10.image = img_mrg_out_10
        Img1OutMerge_10.place(x=250, y=350)
        ##########################################################################################
        # ROC-EVALUATION and DISPLAY
        print("Time for Image Ratioing is %.2f"%(time.time()-t1))
        [fa_list_10, hit_list_10, auroc_10] = GTEval.gt_eval(img_out_10, file_name_3, Tile_size, xy_startend)

        lab_auc_10 = Label(text="IRAT" + spc * 47 + "%.2f" % (abs(auroc_10)), relief=RIDGE)
        lab_auc_10.place(x=715, y=517)

        plt.plot(fa_list_10, hit_list_10, label='Image Ratio', hold=True, linewidth=2)
        fig.canvas.draw()
        ##########################################################################################
    else:
        lab_auc_9 = Label(text="IDIF" + spc * 49 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_9.place(x=715, y=499)

        lab_auc_10 = Label(text="IRAT" + spc * 47 + "%.2f" % (0.00), relief=RIDGE)
        lab_auc_10.place(x=715, y=517)


    # --------------------------------------------------------------------#
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.title('Receiver Operating Characteristics (ROC) curve for threshold parameter')
    plt.legend(bbox_to_anchor=(0.7, 0.4), loc=2, borderaxespad=0.,fontsize=4)
    plt.tick_params(axis='both',which='major',labelsize=7)

    # plt.show()
# -----------------------------------------------------------------------------------------------------------------#

root.minsize(width=950,height=600)
root.title("Multi/Hyper-Spectral CHANGE DETECTION GUI (Python)")

C = Canvas(root,width=950,height=600)
# C.pack(side=LEFT)
Rect_InpImg1 = C.create_rectangle(20,20,220,220,fill='white')  # Input image 1
C.grid(row=0,column=0)
Rect_InpImg2 = C.create_rectangle(250,20,450,220,fill='white')  # Input image 2
C.create_rectangle(480,20,750,244) # around list of algorithms
C.create_rectangle(20,350,220,550,fill='white')  # Output image
C.create_rectangle(250,350,450,550,fill='white')  # Output image merge
roc_curve_disp = C.create_rectangle(480,350,680,550,fill='white')  # ROC curves
C.create_rectangle(710,350,910,550,fill='white')  # List of AUCs
C.create_rectangle(760,20,945,70)  # Tile-window/sliding-window
C.create_rectangle(760,80,945,145)  # Select 'Tile-size'
C.create_rectangle(760,155,945,220)  # Select 'Threshold'

# plot_update = roc_curve_disp.get_tk_widget()

label1 = Label(root,text="INPUT IMAGE: 1",fg="blue")
label1.place(x=20,y=225)
label1 = Label(root,text="INPUT IMAGE: 2",fg="blue")
label1.place(x=250,y=225)
label1 = Label(root,text="CHOOSE ALGORITHM",fg="blue")
label1.place(x=530,y=247)
label1 = Label(root,text="OUTPUT IMAGE",fg="blue")
label1.place(x=70,y=555)
label1 = Label(root,text="OUTPUT IMAGE MERGE",fg="blue")
label1.place(x=290,y=555)
label1 = Label(root,text="ROC CURVES",fg="blue")
label1.place(x=540,y=555)
label1 = Label(root,text="LIST OF AUCs",fg="blue")
label1.place(x=760,y=555)

#********************************************************************************************************************#
def browse_image_1(event):
    #######
    global file_name_1
    file_name_1 = filedialog.askopenfilename()
    img_multi_0 = JJMCIS_Functions.read_envi_img(file_name_1)
    img1_rgb = np.dstack((img_multi_0[:,:,0], img_multi_0[:,:,1], img_multi_0[:,:,2]))
    img1 = Image.fromarray(img1_rgb)
    img1 = img1.resize((200,200),Image.ANTIALIAS)
    img1 = ImageTk.PhotoImage(img1)
    InpImg1 = Label(image=img1)
    InpImg1.image = img1
    InpImg1.place(x=20,y=20)

def browse_image_2(event):
    #######
    global file_name_2
    file_name_2 = filedialog.askopenfilename()
    img_multi_1 = JJMCIS_Functions.read_envi_img(file_name_2)
    img2_rgb = np.dstack((img_multi_1[:, :, 0], img_multi_1[:, :, 1], img_multi_1[:, :, 2]))
    img2 = Image.fromarray(img2_rgb)
    img2 = img2.resize((200, 200), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img2)
    InpImg1 = Label(image=img2)
    InpImg1.image = img2
    InpImg1.place(x=250, y=20)

def browse_image_3(event):
    #######
    global file_name_3
    file_name_3 = filedialog.askopenfilename()

browse_file_1 = Button(C,text="Browse image...",height=2,width=15,fg="white",bg="gray")
browse_file_1.place(x=63,y=93)
browse_file_1.bind("<Button-1>",browse_image_1)
browse_file_2 = Button(C,text="Browse image...",height=2,width=15,fg="white",bg="gray")
browse_file_2.place(x=293,y=93)
browse_file_2.bind("<Button-1>",browse_image_2)
browse_file_3 = Button(C,text="Browse Ground Truth...",height=2,width=18,fg="white",bg="gray")
browse_file_3.place(x=70,y=270)
browse_file_3.bind("<Button-1>",browse_image_3)
####***********########

#********************************************************************************************************************#
##########*************** Dropdown menu for bands ***********###############
band_all_1 = IntVar()
band11 = IntVar()
band21 = IntVar()
band31 = IntVar()
band41 = IntVar()
def check_band_all_1(event):
    val_band1 = band_all_1.get()
    print(val_band1)
    if val_band1==1:
        band11.set(1)
        band21.set(1)
        band31.set(1)
        band41.set(1)
    elif band11.get()==1:
        band11.set(1)
    elif band21.get()==1:
        band21.set(1)
    elif band31.get()==1:
        band31.set(1)
    elif band41.get()==1:
        band41.set(1)
    if (val_band1==0) and (band11.get()==1) and (band21.get()==1) and (band31.get()==1) and (band41.get()==1):
        band11.set(0)
        band21.set(0)
        band31.set(0)
        band41.set(0)

band_all_2 = IntVar()
band12 = IntVar()
band22 = IntVar()
band32 = IntVar()
band42 = IntVar()
#********************************************************************************************************************#
def check_band_all_2(event):
    val_band2 = band_all_2.get()
    # print(val_band2)
    if val_band2==1:
        band12.set(1)
        band22.set(1)
        band32.set(1)
        band42.set(1)
    elif band12.get()==1:
        band12.set(1)
    elif band22.get()==1:
        band22.set(1)
    elif band32.get()==1:
        band32.set(1)
    elif band42.get()==1:
        band42.set(1)
    if (val_band2==0) and (band12.get()==1) and (band22.get()==1) and (band32.get()==1) and (band42.get()==1):
        band12.set(0)
        band22.set(0)
        band32.set(0)
        band42.set(0)

Band1_menu = Menubutton(C,text="SELECT BAND/S",relief=RAISED)
Band1_menu.place(x=120,y=225)
Band2_menu = Menubutton(C,text="SELECT BAND/S",relief=RAISED)
Band2_menu.place(x=350,y=225)
#For image1
Band1_menu.menu = Menu(Band1_menu,tearoff=0)
Band1_menu["menu"]  =  Band1_menu.menu
Band1_menu.menu.add_checkbutton(label="ALL",variable=band_all_1)
Band1_menu.menu.add_checkbutton(label="Band 1",variable=band11)
Band1_menu.menu.add_checkbutton(label="Band 2",variable=band21)
Band1_menu.menu.add_checkbutton(label="Band 3",variable=band31)
Band1_menu.menu.add_checkbutton(label="Band 4",variable=band41)
Band1_menu.bind("<Button-1>",check_band_all_1)
# Default values


#For image2
Band2_menu.menu = Menu(Band2_menu,tearoff=0)
Band2_menu["menu"]  =  Band2_menu.menu
Band2_menu.menu.add_checkbutton(label="ALL",variable=band_all_2)
Band2_menu.bind("<Button-1>",check_band_all_2)
Band2_menu.menu.add_checkbutton(label="Band 1",variable=band12)
Band2_menu.menu.add_checkbutton(label="Band 2",variable=band22)
Band2_menu.menu.add_checkbutton(label="Band 3",variable=band32)
Band2_menu.menu.add_checkbutton(label="Band 4",variable=band42)
# Default values

##########---------------------------------------------------------###############

#********************************************************************************************************************#
##########*************** Algorithm check boxes ***********###############
cb_var1 = IntVar()
cb_var2 = IntVar()
cb_var3 = IntVar()
cb_var4 = IntVar()
cb_var5 = IntVar()
cb_var6 = IntVar()
cb_var7 = IntVar()
cb_var8 = IntVar()
cb_var9 = IntVar()
cb_var10 = IntVar()

def check_alg_all(event):
    val_alg = cb_var1.get()
    print(val_alg)
    if val_alg==0:
        cb_var2.set(1)
        cb_var3.set(1)
        cb_var4.set(1)
        cb_var5.set(1)
        cb_var6.set(1)
        cb_var7.set(1)
        cb_var8.set(1)
        cb_var9.set(1)
        cb_var10.set(1)
    else:
        cb_var2.set(0)
        cb_var3.set(0)
        cb_var4.set(0)
        cb_var5.set(0)
        cb_var6.set(0)
        cb_var7.set(0)
        cb_var8.set(0)
        cb_var9.set(0)
        cb_var10.set(0)

cb_alg1 = Checkbutton(C,text="ALL",variable=cb_var1)
cb_alg1.bind("<Button-1>",check_alg_all)
cb_alg2 = Checkbutton(C,text="POINT DENSITY TAIL ESTIMATION (PDTL)",variable=cb_var2)
cb_alg3 = Checkbutton(C,text="COMPLEXITY/SIMPLEX VOLUME",variable=cb_var3)
cb_alg4 = Checkbutton(C,text="MEAN-SHIFT METRIC",variable=cb_var4)
cb_alg5 = Checkbutton(C,text="OUTLIER-DISTANCE METRIC",variable=cb_var5)
cb_alg6 = Checkbutton(C,text="GRAPH-BASED METHOD",variable=cb_var6)
cb_alg7 = Checkbutton(C,text="CHRONOCHROME CD",variable=cb_var7)
cb_alg8 = Checkbutton(C,text="COVARIANCE EQUALIZATION",variable=cb_var8)
cb_alg9 = Checkbutton(C,text="PCA",variable=cb_var9)
cb_alg10 = Checkbutton(C,text="IMAGE DIFF & IMAGE RATIOING",variable=cb_var10)
cb_alg1.place(x=490,y=21)
cb_alg2.place(x=490,y=43)
cb_alg3.place(x=490,y=65)
cb_alg4.place(x=490,y=87)
cb_alg5.place(x=490,y=109)
cb_alg6.place(x=490,y=131)
cb_alg7.place(x=490,y=153)
cb_alg8.place(x=490,y=175)
cb_alg9.place(x=490,y=197)
cb_alg10.place(x=490,y=219)
##########---------------------------------------------------------###############

#********************************************************************************************************************#
##########*************** Tile-based check box ***********###############
cb_tile_var1 = IntVar()
cb_tile_var2 = IntVar()

def check_tile_all(event):
    val_tile = cb_tile_var1.get()
    if val_tile==0:
        cb_tile_var2.set(0)
    else:
        cb_tile_var2.set(1)
def check_tile_all2(event):
    val_tile2 = cb_tile_var2.get()
    if val_tile2==0:
        cb_tile_var1.set(0)
    else:
        cb_tile_var1.set(1)
cb_tile1 = Checkbutton(C,text="TILE WINDOW",variable=cb_tile_var1)
cb_tile1.bind("<Button-1>",check_tile_all)
cb_tile_var1.set(1)
cb_tile2 = Checkbutton(C,text="SLIDING WINDOW",variable=cb_tile_var2)
cb_tile2.bind("<Button-1>",check_tile_all2)

cb_tile1.place(x=770,y=25)
cb_tile2.place(x=770,y=45)
##########---------------------------------------------------------###############

#********************************************************************************************************************#
##########*************** Tile-size Slider ***********###############
tile_slider = Scale(C,from_=1, to=100,orient=HORIZONTAL,bg='green',fg='white',length=172)
tile_slider.set(20)
tile_slider.place(x=765,y=102)
label_tile = Label(C,text="SELECT TILE-SIZE")
label_tile.place(x=800,y=83)
##########---------------------------------------------------------###############

#********************************************************************************************************************#
##########*************** ADD NOISE Range Slider ***********###############
thresh_slider = Scale(C,from_=0, to=500,orient=HORIZONTAL,bg='blue',fg='white',length=172)
thresh_slider.set(0)
thresh_slider.place(x=765,y=177)
label_thresh = Label(C,text="ADD NOISE: SNR")
label_thresh.place(x=800,y=158)
##########---------------------------------------------------------###############
#********************************************************************************************************************#
RUN_TEST=0
def Run_spotter(Event):
    global NOISE_VAL, TILE_SIZE, TILE_or_SLIDE, ALG_CHECK
    NOISE_VAL = thresh_slider.get()
    print(file_name_1)
    if NOISE_VAL>0:
        ADD_NOISE = True
        print(ADD_NOISE)
        print(NOISE_VAL)
        print("stgartttt")

    TILE_SIZE = tile_slider.get()
    TILE_or_SLIDE = cb_tile_var1.get()
    ALG_CHECK = [cb_var1.get(),cb_var2.get(),cb_var3.get(),cb_var4.get(),cb_var5.get(),cb_var6.get(),cb_var7.get(),
                 cb_var8.get(),cb_var9.get(),cb_var10.get()]
    BAND_IMG_1 = [band11.get(), band21.get(), band31.get(), band41.get()]
    BAND_IMG_2 = [band12.get(), band22.get(), band32.get(), band42.get()]



    print("******* Program Running... *********")
    CD_algorithm()

run_button = Button(C,text="RUN CHANGE DETECTION(CD)",height=2,width=30,fg="white",bg="red")
run_button.bind("<Button-1>",Run_spotter)
run_button.place(y=270,x=390)

root.mainloop()





#****************************************************************#
#********************************************************************************************************************#
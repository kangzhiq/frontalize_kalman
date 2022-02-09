import os
import cv2
import numpy as np
from utils import frames2video

path1 = './MEAD/kalman/'
path2 = './MEAD/RFF/'
path3 = './MEAD/smoothing/'

temp_path = './MEAD/temp/'
file_lst = os.listdir(path1)
for i in range(len(file_lst)):
    filename = file_lst[i]
    img1 = cv2.imread(os.path.join(path1, filename))
    img2 = cv2.imread(os.path.join(path3,filename))
    # img3 = cv2.imread(os.path.join(path3, filename))
    # img4 = cv2.imread(os.path.join(path4,filename))
    # img5 = cv2.imread(os.path.join(path5, filename))
    # img6 = cv2.imread(os.path.join(path6,filename))

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    h, w = img1.shape[:2]

    container = np.zeros((h, 2*w + 20, 3))
    container[:, :w, :]  = img1
    container[:, w+10:2*w+10, :] = img2
    # container[:, 2*w+20:3*w+20, :] = img3
    # container[:, 3*w+30:4*w+30, :] = img4
    # container[:, 4*w+40:5*w+40, :] = img5
    # container[:, 5*w+50:, :] = img6

    cv2.imwrite(os.path.join(temp_path, filename), container)
frames2video('./MEAD/temp/', './MEAD/1_kalman_2_smoo.avi')
#frames2video('./Examples/Obama/temp/', './Examples/Obama/1_init_2_smoo_R_1_3_smoo_R_10_4_smoo_R_100_5_smoo_R_10000_6_smoo_R_10000.avi')


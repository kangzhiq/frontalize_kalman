# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 00:35:30 2020

@author: zhiqi
"""
import numpy as np
import cv2
import face_alignment
from numpy.linalg import inv
import os

from tqdm import tqdm

from BFM.morphable_model import MorphabelModel
import registration_utils as regut
from utils import *


"""
This is the main pipeline of the frontalization, containing the main steps such
as Initalization of parameters, data processing, frontalization of landmarks,
recontruction of frontal view and quantitative evaluation.
"""
def frontalize(profile_path, save_dir=None, save_filename=None, device='cuda',
               region='lip', frontal_path=None, box_save_dir=None, gt_save_dir=None,
               zncc_path=None, visible_only=0, conv_visible=0):
    """
    Main method to run the frontalization pipeline over an input image or video

    Parameters
    ----------
    profile_path : str
        The path to the video/image that is supposed to be frontalized.
    save_dir : str
        The directory to save the frontalized image.
    save_filename : str, optional
        The filename for saving the frontalization image in save_dir. By default,
        the same name as the input image would be used.
    device : str, optional
        The device to be used, either 'cpu' or 'cuda'. By default it is 'cuda'
    region : str, optional
        The targeted region to be frontalized. 'fz' for full size, returning an
        image of the same size as the input with the head centered on the image.
        'lip' for a cropped lip region and 'head' for the cropped head region.
        By default it is 'lip'.
    frontal_path : str, optional
        The path to read the frontal ground truth image, for NCC evaluation. By
        default is None.
    box_save_dir : str, optional
        The path to save the bounding box for running NCC evaluation. By default
        it is None. Only active when the frontal_path is not None.
    gt_save_path : str, optional
        The path to save the ground truth lip region estimated by NCC. As a qualitative
        evaluation. Only active when the frontal_path is not None.
    zncc_path: str, optional
        Path to the file storing the zncc score
    visible_only: int, optional
        Flag indicating the visibility of output image. If 0, the raw transformed
        image will be saved, if 1, the invisible pixels due to self-occlusion
        will be replaecd by black points.

    Returns
    -------
    ROI : list
        A list of frontalized region of interest. The region is determined by
        the region parameter

    """
    #################
    ## Initialization
    #################
    # List of detected landmarks from each initial profile frames
    LMs = []
    # List of forntalized region
    ROI = []
    Masks = []
    # List of frontal ground truth frames
    frontal_frame = []
    # List of inital profile frames to be frontalized
    init_frame = []
    # List of frontal boxes for NCC evaluation
    frontal_boxes = []
    # List of frontalized landmarks
    frontalized_LM_lst = []
    # List of s, R, t to frontalize each frame
    s_lst = []
    R_lst = []
    T_lst = []


    isVideo = True
    if profile_path.split(".")[-1][:3] in ["jpg", "png"]:
        isVideo = False
    if isVideo and profile_path.split(".")[-1][:3] not in ["mp4", "avi"]:
        raise ValueError("File type not supported. Supposeted type: jpg, png, mp4, avi")

    if save_filename is None:
        if isVideo:
            save_filename =  profile_path.split("/")[-1].split(".")[0] + "_frame"
        else:
            save_filename = profile_path.split("/")[-1].split(".")[0]

    if save_dir is None:
        if isVideo:
            save_dir = "./Examples/frontalization_video/"
        else:
            save_dir = "./Examples/frontalization_img/"

    #################
    ## Preparing data
    #################
    ## Loading the BFM model
    bfm = MorphabelModel('BFM/data/BFM.mat')
    sp = bfm.get_shape_para('zero')
    ep = bfm.get_exp_para('zero')
    vertices = bfm.generate_vertices(sp, ep)

    generic_model = vertices[bfm.kpt_ind].copy()
    # Flipping the model to turn it to the standard coordiante system
    generic_model[:, 1:] = - generic_model[:, 1:]
    # Frontal reference model
    LM_ref = generic_model.copy()

    ## Loading video
    if isVideo:
        vidcap = cv2.VideoCapture(profile_path)
        count = 0
        success,image = vidcap.read()
        u_max, v_max = image.shape[:2]
        while success:
            init_frame.append(image)
            success,image = vidcap.read()
            count += 1
    else:
        img = cv2.imread(profile_path)
        if img is None:
            raise ValueError("Image path is wrong, please check again.")
        init_frame.append(img)
        u_max, v_max = img.shape[:2]

    # Extracting landmarks
    # We assume only one person on each frame
    print("Extracting landmarks:")
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device)
    for i in tqdm(range(len(init_frame)), position=0, leave=True):
        LMs.append(fa.get_landmarks(init_frame[i])[0])

    # If frontal ground truth is available
    if frontal_path is not None:
        if isVideo:
            vidcap = cv2.VideoCapture(frontal_path)
            count = 0
            success,image = vidcap.read()
            while success:
                frontal_frame.append(image)
                success,image = vidcap.read()
                count += 1
        else:
            img = cv2.imread(frontal_path)
            frontal_frame.append(img)

        print("Extracting frontal ground truth landmarks:")
        LM_frontal = []
        for i in tqdm(range(len(frontal_frame)), position=0, leave=True):
            LM_frontal.append(fa.get_landmarks(frontal_frame[i])[0])

        ## Get frontal bounding boxes
        LM0 = LM_frontal[0]
        if region == 'lip':
            lip_LM = LM0[48:]
            u, v, nb_u, nb_v = search_region(lip_LM, shape=(1, 1), factor=[2, 1.4])

            # Additional size of box
            V = 25
            H = 25

            box_u = max(0, u-V)
            box_v = max(0, v-H)
            box_nb_u = nb_u+2*V
            box_nb_v = nb_v+2*H

            print("Box region is of size:({}, {})".format(box_nb_u, box_nb_v))

            for i in range(len(frontal_frame)):
                temp = frontal_frame[i][box_u:box_u+box_nb_u, box_v:box_v+box_nb_v,:].astype(int)
                frontal_boxes.append(temp)
                if box_save_dir is not None:
                    cv2.imwrite(box_save_dir + save_filename + "_%04d.jpg" % i, temp)


    #######################
    ## Frontalize landmarks
    #######################
    print("Frontalization of landmarks:")
    maxiter = 200

    for idx in tqdm(range(len(init_frame)), position=0, leave=True):
        M2 = LMs[idx]
        M2[:, 2] = - M2[:, 2]
        if idx == 0:
            R_init, t_init, s_init = regut.compute_align_im(M2.transpose(), LM_ref.transpose())

        R, s, Sig_in, T, w = regut.robust_Student_reg(LM_ref.transpose(), M2.transpose(), R_init, s_init, t_init, maxiter)

        R_init, s_init, t_init = R, s, T

        frontalized_LM = (s* R.dot(M2.transpose()) + T).transpose()
        frontalized_LM_lst.append(frontalized_LM)

        R_lst.append(R)
        s_lst.append(s)
        T_lst.append(T)

    # Rescale the transformation to the natural size
    s_lst = np.array(s_lst)
    T_lst = np.array(T_lst)
    s_max = s_lst.max()
    s_lst /= s_max
    T_lst /= s_max

    frontalized_LM_lst = np.array(frontalized_LM_lst)
    frontalized_LM_lst /= s_max

    ## Move model to the center of image
    frontalized_LM_lst += [[v_max/2, u_max/2, 0] for i in range(68)]
    T_lst += np.array([v_max/2, u_max/2, 0]).reshape(3, 1)

    # LM_ref is generic, using LM_ref to extract lip region cannot cover the exact
    # region. So we use the frontalized landmrks to locate the targeted region
    frontalized_LM = frontalized_LM_lst[0]

    if region == 'fz':
        u = 0
        v = 0
        nb_u = u_max
        nb_v = v_max
    elif region == 'head':
        u, v, nb_u, nb_v = search_region(frontalized_LM, shape=(1, 1), factor=1.5)
    else:
        lip_LM = frontalized_LM[48:]
        u, v, nb_u, nb_v = search_region(lip_LM, shape=(1, 1), factor=[2, 1.4])

    print("Target region is of size:({}, {})".format(nb_u, nb_v))


    #################
    ## Frontalization
    #################
    print("Reconstruction starts:")
    s_3d = None
    R_3d = None
    t_3d = None
    for idx in tqdm(range(len(init_frame)), position=0, leave=True):
        frontalized_LM = frontalized_LM_lst[idx]
        im_profile = init_frame[idx]

        R = R_lst[idx]
        s = s_lst[idx]
        T = T_lst[idx]

        R_prime = inv(R)
        s_prime = 1/s
        T_prime = -s_prime*R_prime.dot(T)

        left_complete = False
        pitch, yaw, roll = matrix2angle(R_prime)
        if yaw < 0 or yaw > 180:
            left_complete = True

        ## 3DMM fitting using frontalized 3D landamrks
        ## The head pose is fixed after the first iteration
        image_vertices, s_3d, R_3d, t_3d = get_3DMM_vertices(frontalized_LM, LM_ref, u_max, v_max, bfm, s_3d, R_3d, t_3d, maxiter = maxiter)
        vis = np.ones(bfm.triangles.shape[0])
        # Z-buffer depth map
        depth_buffer = get_z_map_fast(bfm, u_max, v_max, image_vertices, vis)

        lip_LM = frontalized_LM[48:]
        lip_u, lip_v, lip_nb_u, lip_nb_v = search_region(lip_LM, shape=(1, 1), factor=[2, 1.4])
        # The inner mouth is empty, so we apply the interpolation
#        depth_buffer[lip_u:lip_u+lip_nb_u, lip_v:lip_v+lip_nb_v] = zmap_interpolation(depth_buffer[lip_u:lip_u+lip_nb_u, lip_v:lip_v+lip_nb_v], lip_nb_u, lip_nb_v)
        lip_zbuffer = depth_buffer[u:u+nb_u, v:v+nb_v]

        # Estimate visibility
        if visible_only == 1:
            #vis = get_visibility_normal(bfm, bfm.transform(image_vertices, s_prime, matrix2angle(R_prime), T_prime))
            vis = get_visibility_depth(bfm, u_max, v_max, bfm.transform(image_vertices, s_prime, matrix2angle(R_prime), T_prime))
            # Visibility buffer
            vis_buffer = get_z_map_fast(bfm, u_max, v_max, image_vertices, vis)

            u_head, v_head, nb_u_head, nb_v_head = search_region(frontalized_LM, shape=(1, 1), factor=1)
            if left_complete:
                vis_buffer[u_head:u_head+nb_u_head, v_head:v_head+int(nb_v_head/2)] = zmap_interpolation(vis_buffer[u_head:u_head+nb_u_head, v_head:v_head+int(nb_v_head/2)], nb_u_head, int(nb_v_head/2), invisible_only=1)

            else:
                vis_buffer[u_head:u_head+nb_u_head, v_head+int(nb_v_head/2):v_head+nb_v_head] = zmap_interpolation(vis_buffer[u_head:u_head+nb_u_head, v_head+int(nb_v_head/2):v_head+nb_v_head], nb_u_head, int(nb_v_head/2), invisible_only=1)

            vis_buffer[u:u+nb_u, v:v+nb_v] = neighbor_interpolation(vis_buffer[u:u+nb_u, v:v+nb_v], nb_u, nb_v)
            vis_buffer[lip_u:lip_u+lip_nb_u, lip_v:lip_v+lip_nb_v] = depth_buffer[lip_u:lip_u+lip_nb_u, lip_v:lip_v+lip_nb_v]


        
        ###########################
        ## Transformation
        ###########################
        im_trans = np.zeros((nb_u, nb_v, 3))
        profile_map = np.ones((u_max, v_max)) * np.inf
        profile_index = np.zeros((u_max, v_max))-1
        frontal_map = np.zeros((nb_u, nb_v))
        if region == 'fz':
            idx_list = np.where(depth_buffer != 9999999)
        else:
            idx_list = np.where(lip_zbuffer != 9999999)

        for i, j in zip(idx_list[0], idx_list[1]):
            # recover the depth
            if region == 'fz':
                z_val = depth_buffer[i, j]
            else:
                z_val = lip_zbuffer[i, j]
            # If invisible
            if visible_only == 1 and vis_buffer[i, j] == -9999999:
                im_trans[i, j, :] = [255, 255, 255]
                continue
            m_H = np.array([v+j, u+i, z_val]).reshape(3, 1)
            # Project the pixel from frontal view to the initial image
            pt =  s_prime*R_prime.dot(m_H) + T_prime
            u_prime = int(round(pt[1][0]))
            v_prime = int(round(pt[0][0]))
            if u_prime >= u_max or u_prime < 0 or v_prime >= v_max or v_prime < 0:
                continue
            im_trans[i, j, :] = im_profile[u_prime, v_prime, :]



            if profile_index[u_prime, v_prime] < 0 :
                profile_index[u_prime, v_prime] = i * nb_v + j
                profile_map[u_prime, v_prime] = pt[2][0]
                frontal_map[i, j] = 1
            else:
                if pt[2][0] < profile_map[u_prime, v_prime]:
                    profile_map[u_prime, v_prime] = pt[2][0]
                    pre_u = int(profile_index[u_prime, v_prime] // nb_v)
                    pre_v = int(profile_index[u_prime, v_prime] % nb_v)
                    frontal_map[pre_u, pre_v] = 0
                    profile_index[u_prime, v_prime] = i * nb_v + j
                    frontal_map[i, j] = 1

        # Masks
        frontal_map = neighbor_correct_conv(frontal_map, nb_u, nb_v, 15)
        #frontal_map = neighbor_correct(frontal_map, nb_u, nb_v, 3)
        #frontal_map = neighbor_correct_inv(frontal_map, nb_u, nb_v, 5, ratio=0.3)
        #frontal_map = neighbor_correct_inv(frontal_map, nb_u, nb_v, 25, ratio=0.3, stride=18)

        mask = np.ones((nb_u, nb_v, 3)) * 255
        mask[frontal_map == 0] = [0, 0, 0]
        mask[lip_zbuffer == 9999999] = [255, 255, 255]

        if conv_visible == 1:
            im_trans[mask[:, :, 0] == 0] = [255, 255, 255]
        Masks.append(mask)
        ROI.append(im_trans)
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            cv2.imwrite(save_dir + save_filename + "_%04d.jpg" % idx, im_trans)

    #############
    ## Evaluation
    #############
    # Evaluation is performed only if the frontal ground truth is available
    if frontal_path is not None:
        h, w = ROI[0].shape[:2]

        corr_scores = []
        ground_truth = []
        for i in range(len(ROI)):
            trans_lip = ROI[i]
            frontal_box = frontal_boxes[i]
            gt = LM_frontal[i]

            # Calculate scale
            frontalized = frontalized_LM_lst[i]
            ref_scale = get_scale(gt, frontalized)
            temp_h = int(ref_scale*h)
            temp_w = int(ref_scale*w)
            temp = cv2.resize(trans_lip, (temp_w, temp_h))

            u, v, score = eval_cross_corre(frontal_box, temp)

            corr_scores.append(score)
            im = frontal_box[u:u+h, v:v+w, :]
            ground_truth.append(im)
            if gt_save_dir is not None:
                if not os.path.exists(gt_save_dir):
                    os.makedirs(gt_save_dir)
                cv2.imwrite(gt_save_dir+ save_filename + "_%04d.jpg" % i, im)
        mean_cross_corr = np.array(corr_scores).mean()
        if zncc_path is not None:
            msg = "Input from {}, zncc score:{}\n".format(profile_path, mean_cross_corr)


            my_file = open(zncc_path)
            string_list = my_file.readlines()
            my_file.close()

            my_file = open(zncc_path, "w")
            string_list.append(msg)
            my_file.writelines(string_list)
            my_file.close()
    info = {}
    info['Masks'] = Masks
    frontalized_LM_lst[:, :, :2] -= [v, u]
    info['frontalized_lm'] = frontalized_LM_lst
    info['init_lm'] = LMs
    info['init_frame'] = init_frame
    info['R_lst'] = R_lst
    return ROI, info

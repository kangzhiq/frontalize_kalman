# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 10:59:42 2020

@author: zhiqi
"""
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
import math
from math import cos, sin
from skimage import color
import glob

import registration_utils as regut
from landmarks3D_fitting import *

import    pyximport
pyximport.install()

from BFM.mesh.cython import mesh_core_cython

"""
This file contains the methods utilized during the frontalization procedure, 
such as metric evalation, interpolation, visualization of 3D landmarks, etc. 
"""
def cross_corre(im_C, im_F):
    """
    Caluclate the zero-mean normalized cross correlation of feature window 
    im_F on candidate window im_C.
    
    Parameters
    ----------
    im_C: array-like
        The candidature window on which we would search for the feature region
    im_F: array-like
        The feature window that we are interested in.
        
    Returns
    -------
    scoes: array-like
        An array containing the ZNCC score at each position
        
    Note
    ----
    No padding is performed.
    
    """
    assert im_C.shape[0] >= im_F.shape[0]
    assert im_C.shape[1] >= im_F.shape[1]
    h_F, w_F = im_F.shape[:2]
    if im_C.shape == im_F.shape:
        h = 1
        w = 1
    else:
        h = im_C.shape[0] - im_F.shape[0]
        w = im_C.shape[1] - im_F.shape[1]    
    scores = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            temp = im_C[i:i+h_F, j:j+w_F]
            c = ((temp - temp.mean()) * (im_F - im_F.mean())).sum()
            var_F = ((im_F - im_F.mean())**2).sum()
            var_C = ((temp - temp.mean())**2).sum()
            score = c/np.sqrt(var_F * var_C)
            scores[i, j] = score
    return scores

def eval_cross_corre(im_C, im_F):
    """
    Evaluate the zero-mean cross correlation score between image im_C (larger 
    candidate image) and im_F (smaller feature image) and return the highest
    score and the corresponding position
    
    Parameters
    ----------
    im_C: array-like
        The candidature window on which we would search for the feature region
    im_F: array-like
        The feature window that we are interested in.
        
    Returns
    -------
    y : float
        Column index of the upper-left corner of the corresponding position
    x : float 
        Row index of the upper-left corner of the corresponding position
    max_score : float
        Highest ZNCC score between im_C and im_F
    
    """    
    
    if len(im_C.shape) == 3: im_C = color.rgb2gray(im_C)
    if len(im_F.shape) == 3: im_F = color.rgb2gray(im_F)

    corr = cross_corre(im_C, im_F)
    idx_max = np.argmax(corr)
    y, x = np.unravel_index(idx_max, corr.shape) # find the match
    max_score = corr[y, x]
    return y, x, max_score

def get_scale(gt, frontalized):
    """
    Calculate the scale between frontal reconstruction and ground truth. 
    
    Parameters
    ----------
    gt : array-like
        The list of 68 ground-truth frontal landmarks
    frontalized : 
        The list of 68 frontalized landmarks 
    
    Returns
    -------
    float
        The scale between two set of landmarks
        
    Note
    ----
    We avoid using lip region to make the selected landmarks representative to
    the overall head shape and size.
    """
    # List of landmarks to compare:
    # (45, 36) Outer left eye corner to outer right eye corner
    # (33, 27) Bottom of nose to top of nose
    # (42, 39) Inner left eye corner to inner right eye corner 
    # (16, 0) Upper left jaw to upper right jaw
    # (8, 27) Bottome of jaw to top of nose
    LM_lst = [(45, 36), (33, 27), (42, 39), (16, 0), (8, 27)]        
    nb_LM_lst = len(LM_lst)
    val = 0
    for large, small in LM_lst:
        dist_fr = np.sqrt(((frontalized[large][:2]-frontalized[small][:2])**2).sum())
        dist_gt = np.sqrt(((gt[large][:2]-gt[small][:2])**2).sum())
        val += dist_gt/dist_fr
    return val / nb_LM_lst
        

def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: yaw.
        y: pitch.
        z: roll.
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x, y, z = angles[0], angles[1], angles[2]
    y, x, z = angles[0]/180*np.pi, angles[1]/180*np.pi, angles[2]/180*np.pi

    # x
    Rx=np.array([[1,      0,       0],
                 [0, cos(x),  -sin(x)],
                 [0, sin(x),  cos(x)]])
    # y
    Ry=np.array([[ cos(y), 0, sin(y)],
                 [      0, 1,      0],
                 [-sin(y), 0, cos(y)]])
    # z
    Rz=np.array([[cos(z), -sin(z), 0],
                 [sin(z),  cos(z), 0],
                 [     0,       0, 1]])
    R = Rz.dot(Ry).dot(Rx)
    return R.astype(np.float32)


def matrix2angle(R):
    """
    Transform a rotation matrix to angles of pitch, yaw and roll
    
    Parameters
    ----------
    R : array-like 
        3 x 3 rotation matrix
    
    Returns
    -------
    pitch, yaw, roll: float
        Corresponding angles in degree
    """
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    pitch, yaw, roll = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return pitch, yaw, roll 
 

def cal_MSE(im1, im2):
    """
    Caldulate the Mean Square Error (MSE) between two images
    
    Parameters
    ----------
    im1, im2 : array-like
        Images to compare
    
    Returns
    -------
    float
        MSE value 
    """
    return ((im1 - im2)**2).sum()/np.prod(im1.shape)
    
# MAE = 42.260962818287034
def cal_MAE(im1, im2, idx=None):
    """
    Calculate the Mean Absolute Error (MAE) between two images
    
    Parameters
    ----------
    im1, im2 : array-like
        Images to compare
    
    Returns
    -------
    float
        MAE value 
    """
    return (np.absolute(im1- im2).sum())/np.prod(im1.shape)


def get_SIFT_kp_match(im1, im2):
    """
    Perform SIGT to get corresponding keypoints between two images
    
    Parameters
    ----------
    im1, im2 : array-like
        Images to compare
    
    Returns
    -------
    kp1, kp2 : array-like
        Detected keypoitns using SIFT
    matches : array-like
        The mataching pairs of two set of keypoints
    gray1, gray2 : array-like
        Gray-scale image converted from the initial RGB images
        
    Note
    ----
    This matching is not reliable because the matching keypoints can be highly 
    noisy
    """    
    # convert to grayscale
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Extract descriptor
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    return kp1, kp2, matches, gray1, gray2


def cal_descr_MSE(im1, im2, idx=None):
    """
    Calculate the MSE on matching keypoints detected by SIFT
    
    Parameters
    ----------
    im1, im2 : array-like
        Images to compare
    idx : int, optional
        The index of the image in frame sequence
        
    Returns
    -------
    float
        MSE on detected keypoints
    
    Note
    ----
    distance**2 > 1000 will not be considered
    This metric is not reliable since the matching keypoints are not reliable.
    """
    
    kp1, kp2, matches, gray1, gray2 = get_SIFT_kp_match(im1, im2)
    
    if len(matches) < 5:
        print("Frame{} < 5 keypoints, result not reliable".format(idx))
        return np.inf
    else:
        # Sort them in the order of their distance.
        se = 0
        matches = sorted(matches, key = lambda x:x.distance)    
        count = 0
        for match in matches:
            if count >= 5:
                return se/count
            else:
                pt1 = np.array(kp1[match.queryIdx].pt)
                pt2 = np.array(kp2[match.trainIdx].pt)
                
                dist = ((pt1 - pt2)**2).sum()
                if dist > 500:
                    continue

                se += dist
                count += 1
        if count == 0: return np.inf
        return se/count
        
   
def cal_SSIM(im1, im2):
    """
    Calculate the structural similarity (SSIM) between two images
    
    Parameters
    ----------
    im1, im2 : array-like
        Images to compare
        
    Returns
    -------
    float
        SSIM on detected keypoints
    
    """
    if len(im1.shape) == 3 and len(im2.shape) == 3:
        return ssim(im1, im2, data_range=im2.max()-im2.min(), multichannel=True)
    if len(im1.shape) == 3:
        im1 = color.rgb2gray(im1)
    if len(im2.shape) == 3:
        im2 = color.rgb2gray(im2)
    
    return ssim(im1, im2, data_range=im2.max()-im2.min())


def isPointInTri(point, tri_points):
    """ 
    Judge whether the point is in the triangle.

    Parameters
    ----------
    point : (2,) 
        [u, v] or [x, y] 
    tri_points : (3 vertices, 2 coords) 
        Three vertices(2d points) of a triangle. 
    
    Returns
    -------
    bool
        true for in triangle
    
    Method
    ------
        http://blackpawn.com/texts/pointinpoly/
    """
    tp = tri_points

    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    # check if point in triangle
    return (u >= 0) & (v >= 0) & (u + v < 1)


def get_point_weight(point, tri_points):
    """ 
    
    Parameters
    ----------
        point: (2,). [u, v] or [x, y] 
        tri_points: (3 vertices, 2 coords). three vertices of a triangle. 
    
    Returns
    -------
        w0, w1, w2 : weight of v0, v1, v2
    
    Methods
    -------
        https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
        -m1.compute the area of the triangles formed by embedding the point P inside the triangle
        -m2.Christer Ericson's book "Real-Time Collision Detection". faster.(used)
    """
    tp = tri_points
    # vectors
    v0 = tp[2,:] - tp[0,:]
    v1 = tp[1,:] - tp[0,:]
    v2 = point - tp[0,:]

    # dot products
    dot00 = np.dot(v0.T, v0)
    dot01 = np.dot(v0.T, v1)
    dot02 = np.dot(v0.T, v2)
    dot11 = np.dot(v1.T, v1)
    dot12 = np.dot(v1.T, v2)

    # barycentric coordinates
    if dot00*dot11 - dot01*dot01 == 0:
        inverDeno = 0
    else:
        inverDeno = 1/(dot00*dot11 - dot01*dot01)

    u = (dot11*dot02 - dot01*dot12)*inverDeno
    v = (dot00*dot12 - dot01*dot02)*inverDeno

    w0 = 1 - u - v
    w1 = v
    w2 = u

    return w0, w1, w2


def search_region(LM, shape=None, factor = 1.3, fixed_size = False):
    """
    Given a set of landmarks, search for the bounding box that encapsulate the 
    landmarks
    
    Parameters
    ----------
    LM : array-like
        The set of landmarks that are used to determine the bounding box
    shape : tuple
        The prior knowledge about the region size. The output can be fixed to
        this shape if fixed_size = True
    factor : int, list
        The factor to enlarge the detected bounding box so that it can tolerate 
        some variation of landmarks due to mouth's deformation.
        A list factor contains the enlarge factor for height and width.Otherwise 
        the int factor would be applied to both height and width
    fixed_size : bool
        If true, the shape of the output would be fixed to the shape param.
        
    Returns
    -------
    u, v : int
        The column and row index of the upper-left corner of the bounding box
    nb_u, nb_v : int
        The height and width of the boudning box
    """
    if shape is None:
        shape = (128, 128)
    nb_u, nb_v = shape
    
    if isinstance(factor, list):
        factor_u = factor[0]
        factor_v = factor[1]        
    else:
        factor_u = factor
        factor_v = factor
    
    lips_LM = np.array([(landmark[0], landmark[1]) for landmark in LM], dtype=np.float32)
    
    (v, u, width, height) = cv2.boundingRect(lips_LM)

    # Create a square centered to the mouth and the length is width of mouth
    # u = int(u+height/2-width/2)    
    # nb_u, nb_v = width, width
    if not fixed_size:
        nb_u = int(factor_u*max(nb_u, height))
        nb_v = int(factor_v*max(nb_v, width))    

    #    Resize the region
    if nb_u != height:
        extra_u = nb_u - height
        u = int(u-extra_u/2)
    if nb_v != width:
        extra_v = nb_v - width
        v = int(v-extra_v/2)

    return u, v, nb_u, nb_v


def save_frames(video_path, frame_path):
    """
    Extract frames from a video and save them at the path
    
    Parameters
    ----------
    video_path : str
        The path to read the video
    frame_path : str
        The path to save the extracted frames
    """
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    success,image = vidcap.read()
    while success:
      cv2.imwrite(frame_path + "frame%03d.jpg" % count, image) # save frame as JPEG file
      success,image = vidcap.read()           
      count += 1


def frames2video(frame_path, video_path, img_format = 'jpg'):
    """
    Turn a sequence of frames to a video
    
    Parameters
    ----------
    frame_path : str
        The path to the frames
    video_path : str
        The path to save the video. The video name and the type should be specified
    img_format : str, optional
        The format of the frames
    """
    # Example
    # frame_path = './frames/'
    # video_path = 'test.avi'
    img_array = []
    for filename in sorted(glob.glob(frame_path+'*.'+img_format)):
        img = cv2.imread(filename)        
        img_array.append(img)
        
    if len(img_array) == 0:
        raise ValueError("Path is wrong. Examples of the path: './Examples/frontalization_img/'")

    height, width, layers = img_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 29.97, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("Video saved at {}".format(video_path))
    
    
def compare_3D(p1, p2, color = None):
    """
    Visualize the two sets of 3D landmarks for comparison
    
    Parameters
    ----------
    p1 : array-like
        The first set of 3D landmarks
    p2 : array-like
        The second set of 3D ladnamrks
    color : str, optional
        Specify the color assigned to each set, splitted by " ". Examples:
        color = 'r g' will assign red to the first set and green to the second 
        set. If only one color is given, i.g. "g", two sets will be drawed by
        the same color
    """
    import matplotlib.pyplot as plt
    
    n1 = len(p1)
    n2 = len(p2)
    c1 = 'r'
    c2 = 'k'
    msg = "First point in red, second in black"
    
    if color:
        colors = color.split(" ")
        if len(colors) == 2:
            c1 = colors[0]
            c2 = colors[1]
        else:
            c1 = colors[0]
            c2 = colors[0]
        msg = "Landmarks of the face"
    c = []
    for i in range(n1): c.append(c1)
    for i in range(n2): c.append(c2)
    
    P = np.concatenate((p1, p2))
    
    ax = plt.axes(projection='3d')
    ax.scatter3D(P[:, 0], P[:, 1], P[:, 2], c = c)
    plt.title(msg)
    
    plt.show()


def compare_2D(p1, p2, color=None, filename=None):
    """
    Visualize the two sets of 3D landmarks for comparison

    Parameters
    ----------
    p1 : array-like
        The first set of 3D landmarks
    p2 : array-like
        The second set of 3D ladnamrks
    color : str, optional
        Specify the color assigned to each set, splitted by " ". Examples:
        color = 'r g' will assign red to the first set and green to the second
        set. If only one color is given, i.g. "g", two sets will be drawed by
        the same color
    """
    import matplotlib.pyplot as plt

    n1 = len(p1)
    n2 = len(p2)
    c1 = 'r'
    c2 = 'k'
    if filename is None:
        filename = 'compare_2D.png'
    if color:
        colors = color.split(" ")
        if len(colors) == 2:
            c1 = colors[0]
            c2 = colors[1]
        else:
            c1 = colors[0]
            c2 = colors[0]
        msg = "Landmarks of the face"
    c = []
    for i in range(n1): c.append(c1)
    for i in range(n2): c.append(c2)

    msg = "First point in {}, second in {}".format(c1, c2)

    ax = plt.axes()
    plt.scatter(p1[:, 0], -p1[:, 1], c = c1)
    plt.scatter(p2[:, 0], -p2[:, 1], c = c2)
    #plt.show()
    plt.title(msg)
    plt.savefig(filename)
    plt.close()


def plot_cov(cov, filename='cov.jpg'):
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(19, 15))
    plt.matshow(cov)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=10)
    plt.savefig(filename)
    plt.close()


def get_z_map(bfm, h, w, image_vertices, visibility):
    """
    Get the depth map using the vertices generated from the 3DMM model.
    
    Parameters
    ----------
    bfm : object
        An object of Morphable_model 
    h, w : int
        The hight and width of the rendered image
    image_vertices : array-like
        The vertices generated by BFM model
    
    Returns
    -------
    depth_buffer : array-like
        An array of size h * w containing the depth of each pixel
    
    Note
    ----
    The depth map is initialized with 9999999, indicating the the point is far 
    away from the camera
    
    """
    depth_buffer = np.zeros([h, w]) + 9999999.
    for i in np.where(visibility >= 1)[0]:
        tri = bfm.triangles[i, :] # 3 vertex indices
        
        # the inner bounding box
        umin = max(int(np.ceil(np.min(image_vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(image_vertices[tri, 0]))), w-1)
        
        vmin = max(int(np.ceil(np.min(image_vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(image_vertices[tri, 1]))), h-1)
        
        if umax<umin or vmax<vmin:
            continue
        
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1): 
                if not isPointInTri([u,v], image_vertices[tri, :2]): 
                    continue
                
                w0, w1, w2 = get_point_weight([u, v], image_vertices[tri, :2])
                point_depth = w0*image_vertices[tri[0], 2] + w1*image_vertices[tri[1], 2] + w2*image_vertices[tri[2], 2]
        
                if point_depth < depth_buffer[v, u]:
                    depth_buffer[v, u] = point_depth

    for i in np.where(visibility < 1)[0]:
        tri = bfm.triangles[i, :] # 3 vertex indices
        
        # the inner bounding box
        umin = max(int(np.ceil(np.min(image_vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(image_vertices[tri, 0]))), w-1)
        
        vmin = max(int(np.ceil(np.min(image_vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(image_vertices[tri, 1]))), h-1)
        
        if umax<umin or vmax<vmin:
            continue
        
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], image_vertices[tri, :2]): 
                    continue
                if depth_buffer[v, u] == 9999999.:
                   depth_buffer[v, u] *= -1 
                
    return depth_buffer


def get_z_map_fast(bfm, h, w, image_vertices, visibility):
    """
    Get the depth map using the vertices generated from the 3DMM model.

    Parameters
    ----------
    bfm : object
        An object of Morphable_model
    h, w : int
        The hight and width of the rendered image
    image_vertices : array-like
        The vertices generated by BFM model

    Returns
    -------
    depth_buffer : array-like
        An array of size h * w containing the depth of each pixel

    Note
    ----
    The depth map is initialized with 9999999, indicating the the point is far
    away from the camera

    """
    c = 3
    image = np.zeros((h, w, c), dtype = np.float32)
    depth_buffer = np.zeros([h, w], dtype = np.float32, order = 'C') + 9999999.

    # change orders. --> C-contiguous order(column major)
    vertices = image_vertices.astype(np.float32).copy()
    triangles = bfm.full_triangles.astype(np.int32).copy()
    colors = np.zeros_like(vertices)
    ###
    mesh_core_cython.get_zmap(
                image, vertices, triangles,
                colors,
                depth_buffer,
                vertices.shape[0], triangles.shape[0],
                h, w, c)
    cv2.imwrite("Examples/LRW/frame_{:04d}.jpg".format(1), depth_buffer)
    for i in np.where(visibility < 1)[0]:
        tri = bfm.full_triangles[i, :]  # 3 vertex indices

        # the inner bounding box
        umin = max(int(np.ceil(np.min(image_vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(image_vertices[tri, 0]))), w - 1)

        vmin = max(int(np.ceil(np.min(image_vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(image_vertices[tri, 1]))), h - 1)

        if umax < umin or vmax < vmin:
            continue

        for u in range(umin, umax + 1):
            for v in range(vmin, vmax + 1):
                if not isPointInTri([u, v], image_vertices[tri, :2]):
                    continue
                if depth_buffer[v, u] == 9999999.:
                    depth_buffer[v, u] *= -1

    return depth_buffer


def get_visibility_depth(bfm, h, w, image_vertices):
    # Here h and w should be the height and width of the entire image
    # And the vertices should be transformed to profile.
    depth_buffer = np.zeros([h, w]) + 9999999.
    depth_map = np.zeros([h, w])
    depth_index = np.zeros([h, w])-1
    visible = np.zeros(bfm.triangles.shape[0])

    for i in range(bfm.triangles.shape[0]):
    
        tri = bfm.triangles[i, :] # 3 vertex indices
        
        # the inner bounding box
        umin = max(int(np.ceil(np.min(image_vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(image_vertices[tri, 0]))), w-1)
        
        vmin = max(int(np.ceil(np.min(image_vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(image_vertices[tri, 1]))), h-1)
        
        if umax<umin or vmax<vmin:
            continue
        
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], image_vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], image_vertices[tri, :2])
                point_depth = w0*image_vertices[tri[0], 2] + w1*image_vertices[tri[1], 2] + w2*image_vertices[tri[2], 2]
                depth_map[v, u] += 1
                if point_depth < depth_buffer[v, u]:
                    visible[i] = 1     
                    #print(visible[i])
                    depth_buffer[v, u] = point_depth
                    if depth_index[v, u] != -1:
                        visible[depth_index[v, u].astype(int)] = 0
                    depth_index[v, u] = i
        
    return visible
    
# Assume that all vertices are visible
def get_visibility_depth_inverse(bfm, h, w, image_vertices):
    # Here h and w should be the height and width of the entire image
    # And the vertices should be transformed to profile.
    depth_buffer = np.zeros([h, w]) + 9999999.
    depth_map = np.zeros([h, w])
    depth_index = np.zeros([h, w])-1
    visible = np.ones(bfm.triangles.shape[0])

    for i in range(bfm.triangles.shape[0]):
    
        tri = bfm.triangles[i, :] # 3 vertex indices
        
        # the inner bounding box
        umin = max(int(np.ceil(np.min(image_vertices[tri, 0]))), 0)
        umax = min(int(np.floor(np.max(image_vertices[tri, 0]))), w-1)
        
        vmin = max(int(np.ceil(np.min(image_vertices[tri, 1]))), 0)
        vmax = min(int(np.floor(np.max(image_vertices[tri, 1]))), h-1)
        
        if umax<umin or vmax<vmin:
            continue
        
        tri_visible = 0
        for u in range(umin, umax+1):
            for v in range(vmin, vmax+1):
                if not isPointInTri([u,v], image_vertices[tri, :2]): 
                    continue
                w0, w1, w2 = get_point_weight([u, v], image_vertices[tri, :2])
                point_depth = w0*image_vertices[tri[0], 2] + w1*image_vertices[tri[1], 2] + w2*image_vertices[tri[2], 2]
                depth_map[v, u] += 1
                if point_depth < depth_buffer[v, u]:
                    tri_visible += 1
                    #print(visible[i])
                    depth_buffer[v, u] = point_depth
                    if depth_index[v, u] != -1:
                        visible[depth_index[v, u].astype(int)] -= 1
                    depth_index[v, u] = i
        if tri_visible == 0:
            visible[i] = 0
        else:
            visible[i] += tri_visible
    return visible


def get_visibility_normal(bfm, image_vertices):
    # Here h and w should be the height and width of the entire image
    # And the vertices should be transformed to profile.
    visible = np.ones(bfm.triangles.shape[0])
    
    for i in range(bfm.triangles.shape[0]):
    
        tri = bfm.triangles[i, :] # 3 vertex indices
        
        vertices = image_vertices[tri]
        vec_a = vertices[1] - vertices[0]
        vec_b = vertices[2] - vertices[0]   
        normal = np.cross(vec_a, vec_b)

        cos = np.dot(normal, [0, 0, 1])/np.linalg.norm(normal)
        angle = np.arccos(cos) 
        
        if angle > np.pi/2:
            visible[i] = 0
            
    return visible
    
    
def get_3DMM_vertices(LMs, LM_ref, h, w, bfm, s, R, t, maxiter=200):    
    """
    Fit the 3DMM to the 3D landmarks.
    
    Parameters
    ----------
    LMs : array-like
        The set of targeted landmarks
    LM_ref : array-like
        The set of frontal reference landmarks
    h, w : int
        The height and width of the targeted image
    bfm : object
        An object of Morphabale_Model, containing the data of Basel Face Model
    s, R, t : float, array-like, array-like
        Scaling factor, rotation matrix and translation vector of the rigid
        transformation from inital face to the targeted face.
        If they are not None, then the algorithm will fix them.
    maxiter : int, optional
        Maximum iteration of the fitting procedure
    
    Returns
    -------
    transformed_vertices : array-like
        The set of vertices fitted to the targeted landmarks
    s, R, t : float, array-like, array-like
        Scaling factor. rotation matrix and translation vector. 
    
    Note
    ----
    Here we fix the rigid transformation parameter except the first frame to 
    reduce the shaking effect due to the 3DMM fitting. It is based on the 
    assumption that the frontlaized face should be aligned so the head pose 
    will is supposed to be fixed.
    """
    if R is None:
        # fit
        R_init, t_init, s_init = regut.compute_align_im(LMs.transpose(), LM_ref.transpose())
        R, s, t, ep, sp = robust_Student_reg_3DMM(LMs.transpose(), bfm, R_init, s_init, t_init, maxiter)
        
    else:
        sp, ep = robust_3DMM_given_pose(LMs.transpose(), bfm, R, s, t, maxiter)
    
    fitted_angles = matrix2angle(R)
    fitted_vertices = bfm.generate_vertices(sp, ep)
    transformed_vertices = bfm.transform(fitted_vertices, s, fitted_angles, t)

    return transformed_vertices, s, R, t


def get_3DMM_vertices_smoothing(LMs, LM_ref, h, w, bfm, s, R, t, sp, ep, gamma = 1, maxiter=200):
    """
    Fit the 3DMM to the 3D landmarks.

    Parameters
    ----------
    LMs : array-like
        The set of targeted landmarks
    LM_ref : array-like
        The set of frontal reference landmarks
    h, w : int
        The height and width of the targeted image
    bfm : object
        An object of Morphabale_Model, containing the data of Basel Face Model
    s, R, t : float, array-like, array-like
        Scaling factor, rotation matrix and translation vector of the rigid
        transformation from inital face to the targeted face.
        If they are not None, then the algorithm will fix them.
    maxiter : int, optional
        Maximum iteration of the fitting procedure

    Returns
    -------
    transformed_vertices : array-like
        The set of vertices fitted to the targeted landmarks
    s, R, t : float, array-like, array-like
        Scaling factor. rotation matrix and translation vector.

    Note
    ----
    Here we fix the rigid transformation parameter except the first frame to
    reduce the shaking effect due to the 3DMM fitting. It is based on the
    assumption that the frontlaized face should be aligned so the head pose
    will is supposed to be fixed.
    """
    sp_pre = sp
    ep_pre = ep
    if R is None:
        # fit
        R_init, t_init, s_init = regut.compute_align_im(LMs.transpose(), LM_ref.transpose())
        R, s, t, ep, sp = robust_Student_reg_3DMM(LMs.transpose(), bfm, R_init, s_init, t_init, maxiter)

    else:
        sp, ep = robust_3DMM_given_pose_smoothing(LMs.transpose(), bfm, R, s, t, sp_pre, ep_pre, gamma, maxiter)
        #sp = sp_pre
        #ep = ep_pre


    fitted_angles = matrix2angle(R)
    fitted_vertices = bfm.generate_vertices(sp, ep)
    transformed_vertices = bfm.transform(fitted_vertices, s, fitted_angles, t)

    return transformed_vertices, s, R, t, sp, ep
    
def zmap_interpolation(lip_zbuffer, nb_u, nb_v, invisible_only=0):
    """
    Interpolation of the depth for the lip region, since the 3DMM does not 
    model the inner part of the mouth. 
    
    Parameters
    ----------
    lip_zbuffer : array-like
        The depth map of the lip region
    nb_u, nb_v : int
        The height (nb_u) and width (nb_v) of the lip region.
        
    Returns
    -------
    lip_zbuffer  : array-like
        The lip region with interpolated value
    
    Note
    ----
    This implementation is dedicated to the lip region. For the background 
    region, this algorithm might not work
    """
    # Interpolation
    max_val = 9999999
    # interpolate only invisible pixels
    if invisible_only == 1:
        res = np.where(lip_zbuffer == -max_val)
        lip_zbuffer[lip_zbuffer == -max_val] *= -1
    # Interpolate only background pixels
    elif invisible_only == -1:
        res = np.where(lip_zbuffer == max_val)
        lip_zbuffer[lip_zbuffer == -max_val] *= -1  
    # Interpolate all
    else:
        lip_zbuffer[lip_zbuffer == -max_val] *= -1
        res = np.where(lip_zbuffer == max_val)
    len_res = len(res[0])
    # Reference points for interpolation: left, right, up, down
    ref_pts = np.zeros((len_res, 4)) - 1 
    
    for idx, u, v in zip(range(len_res), res[0], res[1]):
        # Left 
        i, j = u, v
        while j-1 >= 0 and lip_zbuffer[i, j-1] == max_val:
            j -= 1
        if j-1 >= 0: ref_pts[idx, 0] = j-1
        # right
        i, j = u, v
        while j+1 < nb_v and lip_zbuffer[i, j+1] == max_val:
            j += 1
        if j+1 < nb_v: ref_pts[idx, 1] = j+1
        # up
        i, j = u, v
        while i-1 >= 0 and lip_zbuffer[i-1, j] == max_val:
            i -= 1
        if i-1 >= 0: ref_pts[idx, 2] = i-1
        # down
        i, j = u, v
        while i+1 < nb_u and lip_zbuffer[i+1, j] == max_val:
            i += 1
        if i+1 < nb_u: ref_pts[idx, 3] = i+1
        
    # weighted z
    # reference points from left, right, up and down
    for idx, (l, r, u, d) in enumerate(ref_pts.astype(int)):
        i, j = res[0][idx], res[1][idx]
        w_l = 1/(j - l) if l != -1 else 0
        w_r = 1/(r - j) if r != -1 else 0
        w_u = 1/(i - u) if u != -1 else 0
        w_d = 1/(d - i) if d != -1 else 0
        w_sum = w_l + w_r + w_u + w_d
        if w_sum == 0:
            continue
        val   = w_l/w_sum*lip_zbuffer[i, l] +\
                            w_r/w_sum*lip_zbuffer[i, r] +\
                            w_u/w_sum*lip_zbuffer[u, j] +\
                            w_d/w_sum*lip_zbuffer[d, j]
        lip_zbuffer[i, j] = val
    
    if invisible_only == -1:
        lip_zbuffer[lip_zbuffer == max_val] *= -1  
    return lip_zbuffer

def neighbor_interpolation(lip_zbuffer, nb_u, nb_v):
    max_val = max(lip_zbuffer.flatten())
    res = np.where(lip_zbuffer == max_val)
    len_res = len(res[0])
    # Reference points for interpolation: left, right, up, down
    ref_pts = np.zeros((len_res, 4)) - 1 
    
    for idx, u, v in zip(range(len_res), res[0], res[1]):
        #1
        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+1])/4
        #2
        if v-2 >= 0 and lip_zbuffer[u, v-2] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-2]+lip_zbuffer[u, v+1])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+2 < nb_v and lip_zbuffer[u, v+2] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+2])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-2 >= 0 and lip_zbuffer[u-2, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-2, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+1])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+2 < nb_u and lip_zbuffer[u+2, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+2, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+1])/4
        #3
        if v-3 >= 0 and lip_zbuffer[u, v-3] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-3]+lip_zbuffer[u, v+1])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+3 < nb_v and lip_zbuffer[u, v+3] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+3])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-3 >= 0 and lip_zbuffer[u-3, v] != max_val:
                    if u+1 < nb_u and lip_zbuffer[u+1, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-3, v]+lip_zbuffer[u+1, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+1])/4

        if v-1 >= 0 and lip_zbuffer[u, v-1] != max_val: 
            if v+1 < nb_v and lip_zbuffer[u, v+1] != max_val:
                if u-1 >= 0 and lip_zbuffer[u-1, v] != max_val:
                    if u+3 < nb_u and lip_zbuffer[u+3, v] != max_val:
                        lip_zbuffer[u, v] = (lip_zbuffer[u-1, v]+lip_zbuffer[u+3, v]+lip_zbuffer[u, v-1]+lip_zbuffer[u, v+1])/4
    return lip_zbuffer



def neighbor_correct(frontal_map, nb_u, nb_v, kernel_size, ratio=0.6, stride=None):
	kernel_size = kernel_size
	total_num = kernel_size*kernel_size
	half_size = kernel_size // 2
	stride = half_size+1 if stride is None else stride
	for u in range(half_size, nb_u -half_size, stride):
		for v in range(half_size, nb_v -half_size, stride):
			val  = (frontal_map[u-half_size:u+half_size+1, v-half_size:v+half_size+1]).sum()
			if val != total_num and val > total_num*ratio:
				frontal_map[u-half_size:u+half_size+1, v-half_size:v+half_size+1] = 1

	return frontal_map

def neighbor_correct_inv(frontal_map, nb_u, nb_v, kernel_size, ratio=0.6, stride=None):
	kernel_size = kernel_size
	total_num = kernel_size*kernel_size
	half_size = kernel_size // 2
	stride = half_size+1 if stride is None else stride
	for u in range(half_size, nb_u -half_size, stride):
		for v in range(half_size, nb_v -half_size, stride):
			val  = (frontal_map[u-half_size:u+half_size+1, v-half_size:v+half_size+1]).sum()
			if val != total_num and val < total_num*ratio:
				frontal_map[u-half_size:u+half_size+1, v-half_size:v+half_size+1] = 0

	return frontal_map

def neighbor_correct_conv(frontal_map, nb_u, nb_v, kernel_size, ratio=0.6, stride=None):
	import torch
	h, w = frontal_map.shape[:2]
	frontal_map = torch.tensor(frontal_map.reshape(1, 1, nb_u, nb_v))
	
	with torch.no_grad():
		Conv = torch.nn.Conv2d(1, 1, kernel_size, padding = (kernel_size // 2), bias=False)
		Conv.weight.fill_(1 / (kernel_size * kernel_size))
		temp = Conv(frontal_map.float())

	frontal_map = (temp > 0.6).float()	
	return frontal_map.cpu().detach().numpy().reshape((nb_u, nb_v))
	


def erode_frontal_map(frontal_map, dist):
	nb_u, nb_v = frontal_map.shape[:2]
		  







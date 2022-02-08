# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:09:01 2020

@author: zhiqi
"""
import numpy as np
from registration_utils import *

"""
This file contains the methods for 3DMM fitting using 3D landmarks
"""
def estimate_shape(x, shapeMU, shapePC, shapeEV, expression, s, R, t2d, lamb = 3000):
    """
    Estimate the shape parameters for 3DMM fitting. Since it is a linear system,
    we have a closed-form solution for the parameter estiamtion.
    
    Parameters
    ----------
    x : array-like, (2, n). 
        Image points (to be fitted)
    shapeMU : array-like,  (3n, 1)
        Mean shape of the 3DMM 
    shapePC : array-like, (3n, n_sp)
        Principal component of the shape PCA   
    shapeEV : array-like, (n_sp, 1)
        The eigenvalue of the shape PCA    
    expression: (3, n)
        Expression variation    
    s, R, t2d : float, array-like, array-like
        The rigid transformtion estimated by 3DMM
    lambda : int
        Regulation coefficient

    Returns
    -------
    shape_para : array-like, (n_sp, 1) 
        The shape parameters(coefficients)
    """
    x = x.copy()
    assert(shapeMU.shape[0] == shapePC.shape[0])
    assert(shapeMU.shape[0] == x.shape[1]*3)

    dof = shapePC.shape[1]

    n = x.shape[1]
    sigma = shapeEV
    t2d = np.array(t2d)
    A = s*R

    # --- calc pc
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 199 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T.copy()) # 199 x n x 2
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 199

    # --- calc b
    # shapeMU
    mu_3d = np.resize(shapeMU, [n, 3]).T # 3 x n
    # expression
    exp_3d = expression
    # 
    b = A.dot(mu_3d + exp_3d) + t2d # 2 x n
    b = np.reshape(b.T, [-1, 1]) # 2n x 1
    # --- solve
    equation_left = np.dot(pc.T, pc) + lamb * np.diagflat(1/sigma**2)
    x = np.reshape(x.T, [-1, 1])
    equation_right = np.dot(pc.T, x - b)

    shape_para = np.dot(np.linalg.inv(equation_left), equation_right)

    return shape_para


def estimate_non_rigid_param_3D_robust(M, shapeMU, PC, EV, non_rigid, s, R, t, w, invSig):
    """
    Estimate the non rigid parameters, namely shape or expression parameters,
    for 3DMM fitting. An generalization of estimate_shape and estimate_expression
    methods.
    
    Parameters
    ----------
    M : array-like, (2, n). 
        Landmarks to be fitted
    shapeMU : array-like,  (3n, 1)
        Mean shape of the 3DMM 
    PC : array-like
        Principal component of the PCA   
    EV : array-like
        The eigenvalue of the PCA    
    non_rigid : (3, n)
        The non-rigid variation of another source.    
    s, R, t2d : float, array-like, array-like
        The rigid transformtion estimated by 3DMM
    w : array-like
        The weight parameters estimated by the robust student method
    invSig : array-like
        The inverse of the covariance matrix

    Returns
    -------
    alpha : array-like 
        The non-rigid parameters
        
    Note
    ----
    If we are estimating the shape parameters, then the non-rigid variation by
    expression will be presented by the non_rigid parameters, and vise-versa.
    """
    M = M.copy()
    assert(shapeMU.shape[0] == PC.shape[0])
    assert(shapeMU.shape[0] == M.shape[1]*3)

    dof = PC.shape[1]
    n = M.shape[1]
    t = t.reshape((3, 1))


    # --- calc pc
    left = np.zeros((dof, dof))
    right = np.zeros((dof, 1))
    for i in range(n):
        F = s * R @ PC[3*i: 3*i+3, :] # 3 x 199
        v_mean = shapeMU[3*i: 3*i+3].reshape(3, 1)
        v_prime = s * R @ (v_mean + non_rigid[:, i].reshape(3, 1)) + t
        e = M[:, i].reshape((3, 1)) - v_prime
        e.reshape((3, 1))
        left += w[0][i] * (F.T @ invSig @ F)
        right += w[0][i] * F.T @ invSig @ e
     
    # --- solve
    alpha = np.linalg.inv(left) @ right

    return alpha  


def estimate_non_rigid_param_3D_robust_regu(M, shapeMU, PC, EV, non_rigid, s, R, t, w, invSig, lamb = 100):
    """
    Implementation of estimate_non_rigid_param_3D_robust with regularization 
    term.
    
    Parameters
    ----------
    M : array-like, (2, n). 
        Landmarks to be fitted
    shapeMU : array-like,  (3n, 1)
        Mean shape of the 3DMM 
    PC : array-like
        Principal component of the PCA   
    EV : array-like
        The eigenvalue of the PCA    
    non_rigid : (3, n)
        The non-rigid variation of another source.    
    s, R, t2d : float, array-like, array-like
        The rigid transformtion estimated by 3DMM
    w : array-like
        The weight parameters estimated by the robust student method
    invSig : array-like
        The inverse of the covariance matrix
    lamb : int, optional
        The regularization parameter
    
    Returns
    -------
    alpha : array-like 
        The non-rigid parameters
        
    Note
    ----
    If we are estimating the shape parameters, then the non-rigid variation by
    expression will be presented by the non_rigid parameters, and vise-versa.
    """
    M = M.copy()
    assert(shapeMU.shape[0] == PC.shape[0])
    assert(shapeMU.shape[0] == M.shape[1]*3)

    dof = PC.shape[1]
    n = M.shape[1]
    t = t.reshape((3, 1))


    # --- calc pc
    left = np.zeros((dof, dof))
    right = np.zeros((dof, 1))
    for i in range(n):
        F = s * R @ PC[3*i: 3*i+3, :] # 3 x 199
        v_mean = shapeMU[3*i: 3*i+3].reshape(3, 1)
        v_prime = s * R @ (v_mean + non_rigid[:, i].reshape(3, 1)) + t
        e = M[:, i].reshape((3, 1)) - v_prime
        e.reshape((3, 1))
        left += w[0][i] * (F.T @ invSig @ F)
        right += w[0][i] * F.T @ invSig @ e
        
    # Regularization term based on variance of each PC
    left += lamb * np.diagflat(1/EV**2)
     
    # --- solve
    alpha = np.linalg.inv(left) @ right

    return alpha  


def objectif_function(Y, X, s, R, t, sigma, w, alpha_sp, EV_sp, lamb_sp, alpha_ep, EV_ep, lamb_ep):
    """
    The objectif function of the optimization problem during the 3DMM fitting 
    procedure.
    
    Parameters
    ----------
    Y : array-like
        Targeted points to be fitted on
    X : array-like
        Initial starting points to be fitted to Y.
    s, R, t : float, array-like, array-like
        Scaling factor, rotation matix and translation vector of the rigid 
        transformation between X and Y
    sigma : array-like
        Covariance matrix
    w : array-like
        The weight parameters from robust student estimation
    alpha_sp, alpha_ep : array-like
        The non-rigid parameter for shape and expression variation
    EV_sp, EV_ep : array-like
        The Eigenvalue of shape and expression
    lamb_sp, lamb_ep : int
        Regularization parameter for shape fitting and expression fitting
    
    Returns
    -------
    float
        The value of the objectif function
    """
    n = X.shape[1]
    log_sig = np.log(np.sqrt((sigma**2).sum()))
    regu_sp = lamb_sp * alpha_sp.T @ np.diagflat(1/EV_sp**2) @ alpha_sp
    regu_ep = lamb_ep * alpha_ep.T @ np.diagflat(1/EV_ep**2) @ alpha_ep
    diff = ((Y - s* R @ X - t)**2).sum(axis=0) @ w.reshape((-1, 1)) 
    return diff + log_sig * n  + regu_sp + regu_ep

def enlarge_weights(w):
    w[0, 60:68] *= 5
    w[0, :17] *= 5
    return w

def robust_Student_reg_3DMM(Y, bfm, R0, s0, t0, maxiter=200):
    """
    An implementation of robust_Student_reg method with 3DMM fitting integrated.
    The idea is to iteratively update the pose parameters and shape/expression
    parameters for 3DMM fitting using EM algorithm.
    
    Parameters
    ----------
    Y : array-like
        Targeted points for 3DMM to fit
    bfm : object
        An instance of Morphable_Model encapsulating the data of Basel Face Model
    R0, s0, t0 : array-like, float, array-like
        Rigid tranformation parameters for initialization
    maxiter : int, optional
        Maximum number of iteration for 3DMM fitting
        
    Returns
    -------
    R, s, t : array-like, float, array-like
        Estimated rigid transformation from 3DMM mean face to the targeted points
    ep : array-like
        Expression parameters for 3DMM fitting
    sp : array-like
        Shape parameters for 3DMM fitting    
    """
    
    model = bfm.model
    n_ep = bfm.n_exp_para
    n_sp = bfm.n_shape_para
    X_ind = bfm.kpt_ind
    lamb = 100

    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)
    
    #-------------------- estimate
    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    
    # generate initial vertices and landmarks
    vertices = bfm.generate_vertices(sp, ep)
    X = vertices[X_ind, :3].T.copy()

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
    
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    D, N = X.shape
    s = s0
    R = R0 
    
    q = quaternion_from_matrix(R)
    
    A = Yc - s*R.dot(Xc)

    Sig_in = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    invSig = np.linalg.inv(Sig_in)
        
    nu=1*np.ones((1,N))
    mu=1*np.ones((1,N))
    
    loss_old = np.inf
        
    for iter in range(maxiter):
        
        # E-step:
        a, b, w = E_step_Student(Yc-s*R.dot(Xc), mu, nu, invSig, N)
        w  = np.reshape(w,(1,-1))
        assert(w.shape==(1,N))
        Xc, Yc, Xb, Yb = center_datas(X,Y,w,N)

                    
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= float(w[0,i])
                
            Sig_in = A.dot(A.T) + 1e1*np.eye(D)
            Sig_in /= N
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)
    

            for i in range(N):
                mu[0,i] = invpsi(psi(a[0,i])-1./N*sum([math.log(b[0,j]) for j in range(N)]))            
            s = compute_scale(Xc,Yc,R,invSig,w)

            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,w,N,q,opti="SLSQP_JAC",reg_param=10)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
            
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)

        t = Yb - s*R.dot(Xb)
        t = np.reshape(t,(3,1))
            

        #----- estimate shape
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        w = enlarge_weights(w)
        ep = estimate_non_rigid_param_3D_robust_regu(Y, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t, w, invSig, lamb)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        sp = estimate_non_rigid_param_3D_robust_regu(Y, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t, w, invSig, lamb)

        
        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T     
        
        loss = objectif_function(Y, X, s, R, t, Sig_in, w, sp.reshape((-1, 1)), model['shapeEV'][:n_sp,:], lamb, ep.reshape((-1, 1)), model['expEV'][:n_ep,:], lamb)

        
        if iter > 0:
            ratio = (loss - loss_old)/loss_old
            sign = np.sign(ratio)
            if ratio*sign < 0.0001:
                break

        loss_old = loss    
         
    return R, s, t, ep, sp


def estimate_non_rigid_param_3D_robust_regu_no_weight(M, shapeMU, PC, EV, non_rigid, s, R, t, lamb = 100):
    """
    Implementation of estimate_non_rigid_param_3D_robust_regu without weights.
    
    Parameters
    ----------
    M : array-like, (2, n). 
        Landmarks to be fitted
    shapeMU : array-like,  (3n, 1)
        Mean shape of the 3DMM 
    PC : array-like
        Principal component of the PCA   
    EV : array-like
        The eigenvalue of the PCA    
    non_rigid : (3, n)
        The non-rigid variation of another source.    
    s, R, t2d : float, array-like, array-like
        The rigid transformtion estimated by 3DMM
    lamb : int, optional
        The regularization parameter
    
    Returns
    -------
    alpha : array-like 
        The non-rigid parameters
        
    Note
    ----
    If we are estimating the shape parameters, then the non-rigid variation by
    expression will be presented by the non_rigid parameters, and vise-versa.
    """
    M = M.copy()
    assert(shapeMU.shape[0] == PC.shape[0])
    assert(shapeMU.shape[0] == M.shape[1]*3)

    dof = PC.shape[1]
    n = M.shape[1]
    t = t.reshape((3, 1))


    # --- calc pc
    left = np.zeros((dof, dof))
    right = np.zeros((dof, 1))
    for i in range(n):
        F = s * R @ PC[3*i: 3*i+3, :] # 3 x 199
        v_mean = shapeMU[3*i: 3*i+3].reshape(3, 1)
        v_prime = s * R @ (v_mean + non_rigid[:, i].reshape(3, 1)) + t
        e = M[:, i].reshape((3, 1)) - v_prime
        e.reshape((3, 1))
        left += F.T @ F
        right += F.T @ e
        
    # Regularization term based on variance of each PC
    left += lamb * np.diagflat(1/EV**2)
     
    # --- solve
    alpha = np.linalg.inv(left) @ right

    return alpha


def estimate_non_rigid_param_3D_robust_regu_no_weight_smoothing(M, shapeMU, PC, EV, non_rigid, s, R, t, alpha_pre, gamma = 0.01, lamb=100):
    """
    Implementation of estimate_non_rigid_param_3D_robust_regu without weights.

    Parameters
    ----------
    M : array-like, (2, n).
        Landmarks to be fitted
    shapeMU : array-like,  (3n, 1)
        Mean shape of the 3DMM
    PC : array-like
        Principal component of the PCA
    EV : array-like
        The eigenvalue of the PCA
    non_rigid : (3, n)
        The non-rigid variation of another source.
    s, R, t2d : float, array-like, array-like
        The rigid transformtion estimated by 3DMM
    lamb : int, optional
        The regularization parameter

    Returns
    -------
    alpha : array-like
        The non-rigid parameters

    Note
    ----
    If we are estimating the shape parameters, then the non-rigid variation by
    expression will be presented by the non_rigid parameters, and vise-versa.
    """
    M = M.copy()
    assert (shapeMU.shape[0] == PC.shape[0])
    assert (shapeMU.shape[0] == M.shape[1] * 3)

    dof = PC.shape[1]
    n = M.shape[1]
    t = t.reshape((3, 1))

    # --- calc pc
    left = np.zeros((dof, dof))
    right = np.zeros((dof, 1))
    for i in range(n):
        F = s * R @ PC[3 * i: 3 * i + 3, :]  # 3 x 199
        v_mean = shapeMU[3 * i: 3 * i + 3].reshape(3, 1)
        v_prime = s * R @ (v_mean + non_rigid[:, i].reshape(3, 1)) + t
        e = M[:, i].reshape((3, 1)) - v_prime
        e.reshape((3, 1))
        left += F.T @ F
        right += F.T @ e

    # Regularization term based on variance of each PC

    left += lamb * np.diagflat(1 / EV ** 2)
    # Regularization of the shape parameters
    left += gamma * np.diagflat(np.ones_like(EV))

    # Regularization of the shape parameters
    right += gamma * alpha_pre
    # --- solve
    alpha = np.linalg.inv(left) @ right

    return alpha


def objectif_function_no_weight(Y, X, s, R, t, alpha_sp, EV_sp, lamb_sp, alpha_ep, EV_ep, lamb_ep):
    """
    Implementation of the objectif_function without weight parameters.
    
    Parameters
    ----------
    Y : array-like
        Targeted points to be fitted on
    X : array-like
        Initial starting points to be fitted to Y.
    s, R, t : float, array-like, array-like
        Scaling factor, rotation matix and translation vector of the rigid 
        transformation between X and Y
    alpha_sp, alpha_ep : array-like
        The non-rigid parameter for shape and expression variation
    EV_sp, EV_ep : array-like
        The Eigenvalue of shape and expression
    lamb_sp, lamb_ep : int
        Regularization parameter for shape fitting and expression fitting
    
    Returns
    -------
    float
        The value of the objectif function
    """    
    regu_sp = lamb_sp * alpha_sp.T @ np.diagflat(1/EV_sp**2) @ alpha_sp
    regu_ep = lamb_ep * alpha_ep.T @ np.diagflat(1/EV_ep**2) @ alpha_ep
    diff = ((Y - s* R @ X - t)**2).sum()  
    return diff + regu_sp + regu_ep


def robust_3DMM_given_pose(Y, bfm, R, s, t, maxiter=200):
    """
    Fitting 3DMM with a given pose for the model. Namely estimating the shape 
    and expression parameters.
    
    Parameters
    ----------
    Y : array-like
        Targeted points for 3DMM to fit
    bfm : object
        An instance of Morphable_Model encapsulating the data of Basel Face Model
    R, s0 t : array-like, float, array-like
        Rigid tranformation parameters for the pose.
    maxiter : int, optional
        Maximum number of iteration for 3DMM fitting
        
    Returns
    -------
    ep : array-like
        Expression parameters for 3DMM fitting
    sp : array-like
        Shape parameters for 3DMM fitting    
    """
    model = bfm.model
    n_ep = bfm.n_exp_para
    n_sp = bfm.n_shape_para
    X_ind = bfm.kpt_ind
    lamb = 100

    # Initialization of shape and expression parameters
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)

    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1])*3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    loss_old = np.inf
        
    for iter in range(maxiter):
    
        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape)/3), 3]).T
        ep = estimate_non_rigid_param_3D_robust_regu_no_weight(Y, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t)
    
        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        sp = estimate_non_rigid_param_3D_robust_regu_no_weight(Y, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t)

        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X)/3), 3]).T     
        loss = objectif_function_no_weight(Y, X, s, R, t, sp.reshape((-1, 1)), model['shapeEV'][:n_sp,:], lamb, ep.reshape((-1, 1)), model['expEV'][:n_ep,:], lamb)
#        print("Loss: ", loss)
        
        if iter > 0:
            ratio = (loss - loss_old)/loss_old
            sign = np.sign(ratio)
            if ratio*sign < 0.0001:
                break

        loss_old = loss           
    return sp, ep


def robust_3DMM_given_pose_smoothing(Y, bfm, R, s, t, sp_pre, ep_pre, gamma=1, maxiter=200):
    """
    Fitting 3DMM with a given pose for the model. Namely estimating the shape
    and expression parameters.

    Parameters
    ----------
    Y : array-like
        Targeted points for 3DMM to fit
    bfm : object
        An instance of Morphable_Model encapsulating the data of Basel Face Model
    R, s0 t : array-like, float, array-like
        Rigid tranformation parameters for the pose.
    maxiter : int, optional
        Maximum number of iteration for 3DMM fitting

    Returns
    -------
    ep : array-like
        Expression parameters for 3DMM fitting
    sp : array-like
        Shape parameters for 3DMM fitting
    """
    model = bfm.model
    n_ep = bfm.n_exp_para
    n_sp = bfm.n_shape_para
    X_ind = bfm.kpt_ind
    lamb = 100

    # Initialization of shape and expression parameters
    sp = sp_pre
    ep = ep_pre

    X_ind_all = np.tile(X_ind[np.newaxis, :], [3, 1]) * 3
    X_ind_all[1, :] += 1
    X_ind_all[2, :] += 2
    valid_ind = X_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    loss_old = np.inf

    for iter in range(maxiter):

        # expression
        shape = shapePC.dot(sp)
        shape = np.reshape(shape, [int(len(shape) / 3), 3]).T
        ep = estimate_non_rigid_param_3D_robust_regu_no_weight_smoothing(Y, shapeMU, expPC, model['expEV'][:n_ep, :],
                                                                         shape, s, R, t, ep_pre, gamma)

        # shape
        expression = expPC.dot(ep)
        expression = np.reshape(expression, [int(len(expression) / 3), 3]).T
        sp = estimate_non_rigid_param_3D_robust_regu_no_weight_smoothing(Y, shapeMU, shapePC, model['shapeEV'][:n_sp, :],
                                                               expression, s, R, t, sp_pre, gamma)

        X = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
        X = np.reshape(X, [int(len(X) / 3), 3]).T
        loss = objectif_function_no_weight(Y, X, s, R, t, sp.reshape((-1, 1)), model['shapeEV'][:n_sp, :], lamb,
                                           ep.reshape((-1, 1)), model['expEV'][:n_ep, :], lamb)
        #        print("Loss: ", loss)

        if iter > 0:
            ratio = (loss - loss_old) / loss_old
            sign = np.sign(ratio)
            if ratio * sign < 0.0001:
                break

        loss_old = loss
    return sp, ep
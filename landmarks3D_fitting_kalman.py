# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 17:09:01 2020

@author: zhiqi
"""
import numpy as np
from registration_utils import * 
import registration_utils as regut
from utils import angle2matrix, compare_2D

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
    pc_3d = np.resize(shapePC.T, [dof, n, 3]) # 29 x n x 3
    pc_3d = np.reshape(pc_3d, [dof*n, 3]) 
    pc_2d = pc_3d.dot(A.T.copy()) # 29 x n x 2
    
    pc = np.reshape(pc_2d, [dof, -1]).T # 2n x 29

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
    #print('mean value: {}'.format(np.mean(left)))
    #print('mean regu: {}'.format(np.mean(lamb * np.diagflat(1/EV**2))))           
    # Regularization term based on variance of each PC
    left += lamb * np.diagflat(1/EV**2)
     
    # --- solve
    alpha = np.linalg.inv(left) @ right

    return alpha  


def estimate_non_rigid_param_3D_robust_regu_kalman(M, shapeMU, PC, EV, non_rigid, s, R, t, w, invSig, lamb = 100):
    """
    Implementation of estimate_non_rigid_param_3D_robust with regularization 
    term.
    
    Parameters
    ----------
    M : array-like, (3, n). 
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
        F = PC[3*i: 3*i+3, :] # 3 x 29
        v_mean = shapeMU[3*i: 3*i+3].reshape(3, 1)
        v_prime = v_mean + non_rigid[:, i].reshape(3, 1)
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
    diff = ((Y - s* R @ X - t)**2).sum(axis=0)  @ w.reshape((-1, 1)) 
    diff /= np.mean(Y)**2
    # print('loss: diff: {}, log_sig: {}, sp: {}, ep: {}'.format(diff, log_sig*n, regu_sp, regu_ep))
    return diff + log_sig * n # + regu_sp + regu_ep

def enlarge_weights(w):
    w[0, 60:68] *= 5
    w[0, :17] *= 5
    return w

# Here we calculate the shape and pose together
def compute_init(X, bfm, R0, s0, t0, maxiter=200):    
    model = bfm.model
    n_ep = bfm.n_exp_para
    n_sp = bfm.n_shape_para
    V_ind = bfm.kpt_ind
    lamb = 100

    #-- init
    sp = np.zeros((n_sp, 1), dtype = np.float32)
    ep = np.zeros((n_ep, 1), dtype = np.float32)
    
    #-------------------- estimate
    V_ind_all = np.tile(V_ind[np.newaxis, :], [3, 1])*3
    V_ind_all[1, :] += 1
    V_ind_all[2, :] += 2
    valid_ind = V_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    # Rotate the vertices 180 degree around x-axis
    shapeMU = np.reshape(shapeMU, [int(3), int(len(shapeMU)/3)], 'F').T
    #shapeMU = np.reshape(shapeMU, [int(len(shapeMU)/3), 3])
    shapeMU[:, 1:] = -shapeMU[:, 1:]

    #V = np.reshape(shapeMU, [int(len(shapeMU)/3), 3]).T.copy()
    V = shapeMU.T.copy()
    shapeMU = shapeMU.reshape((68*3, 1))


    # X[1:, :] = - X[1:, :]
    # X, V: D*N
    Xc = X.copy()
    Vc = V.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Vc -= np.mean(V, axis = 1, keepdims = True)

    # from utils import compare_2D
    # compare_2D(Xc[:, :40].T, Vc[:, :40].T/300, filename="./verify/00{}test.jpg".format(0))
    
    D, N = X.shape
    s = s0
    R = R0 
    
    q = quaternion_from_matrix(R)
    
    A = Vc - s*R.dot(Xc)

    Sig_in = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    invSig = np.linalg.inv(Sig_in)
        
    nu=1*np.ones((1,N))
    mu=1*np.ones((1,N))
    
    loss_old = np.inf

    from utils import matrix2angle
    print("new algo init: R:{}, s:{}".format(matrix2angle(R), s))    
    for iter in range(maxiter):
        
        # E-step:
        a, b, w = E_step_Student(Vc-s*R.dot(Xc), mu, nu, invSig, N)
        w  = np.reshape(w,(1,-1))
        assert(w.shape==(1,N))
        Xc, Vc, Xb, Vb = center_datas(X,V,w,N)
        
        # update Sig-in:
        A = Vc - s*R.dot(Xc)
        for i in range(N):
            A[:,i] *= float(w[0,i])
            
        Sig_in = A.dot(A.T) + 1e1*np.eye(D)
        Sig_in /= N
        
        # update scale:
        invSig = np.linalg.inv(Sig_in)


        for i in range(N):
            mu[0,i] = invpsi(psi(a[0,i])-1./N*sum([math.log(b[0,j]) for j in range(N)]))            
        s = compute_scale(Xc,Vc,R,invSig,w)

        q,A,B           = min_rotation_over_q(Xc,Vc,s,invSig,w,N,q,opti="SLSQP_JAC",reg_param=10)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
        

        q               = q/np.linalg.norm(q) #TODO check if q is unit
        R               = rot(q)

        t = Vb - s*R.dot(Xb)
        t = np.reshape(t,(3,1))
            
        Y = s*R.dot(X) + t
        # compare_2D(Y.T, Vc.T, filename="./verify/00{}fitted.jpg".format(iter))
        # print("new algo: R:{}, s:{}, t:{}, iter: {}".format(matrix2angle(R), s, t, iter))
        #----- estimate shape
        # Allows for more iterations for shape
        for i in range(1):
            # expression
            shape = shapePC.dot(sp)
            shape = np.reshape(shape, [int(len(shape)/3), 3]).T
            #w = enlarge_weights(w)
            ep = estimate_non_rigid_param_3D_robust_regu_kalman(Y, shapeMU, expPC, model['expEV'][:n_ep,:], shape, s, R, t, w, invSig, lamb)

            a = shapeMU
            a = np.reshape(a, [68, 3])
            b = shapeMU + expPC.dot(ep)
            b = np.reshape(b, [68, 3])
            
            # shape
            expression = expPC.dot(ep)
            expression = np.reshape(expression, [int(len(expression)/3), 3]).T
            sp = estimate_non_rigid_param_3D_robust_regu_kalman(Y, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t, w, invSig, lamb*100)

            V_fitting = shapeMU + shapePC.dot(sp) + expPC.dot(ep)
            V_fitting = np.reshape(V_fitting, [int(len(V_fitting)/3), 3]).T   

            V = V_fitting
            Vc = V - np.mean(V, axis = 1, keepdims = True)          
            

        loss = objectif_function(V_fitting, X, s, R, t, Sig_in, w, sp.reshape((-1, 1)), model['shapeEV'][:n_sp,:], lamb, ep.reshape((-1, 1)), model['expEV'][:n_ep,:], lamb)
        # print('iter: {} shape loss: {}'.format(iter, loss))
        # if iter % 5 == 0:
        #     compare_2D(V_fitting.transpose(), Y.transpose(), filename="./verify/lm{}.jpg".format(iter))
        
        if iter > 0:
            ratio = (loss - loss_old)/loss_old
            sign = np.sign(ratio)
            if ratio*sign < 0.0001:
                break

        loss_old = loss    
        V = V_fitting
        Vc = V - np.mean(V, axis = 1, keepdims = True)
        # compare_2D(Y.T, Vc.T, filename="./verify/00{}fitted.jpg".format(iter))
    # print('weight: {}'.format(w))     
    return R, s, t, ep, sp, Sig_in, V_fitting


def compute_shape(X, R, s, Sig_in, t, bfm, Gamma_s, Gamma_v, v, Psi, P, shape_param, alpha):
    model = bfm.model
    n_ep = bfm.n_exp_para
    n_sp = bfm.n_shape_para
    V_ind = bfm.kpt_ind
    lamb = 100

    #-------------------- estimate
    V_ind_all = np.tile(V_ind[np.newaxis, :], [3, 1])*3
    V_ind_all[1, :] += 1
    V_ind_all[2, :] += 2
    valid_ind = V_ind_all.flatten('F')

    shapeMU = model['shapeMU'][valid_ind, :]
    shapePC = model['shapePC'][valid_ind, :n_sp]
    expPC = model['expPC'][valid_ind, :n_ep]

    # Rotate the vertices 180 degree around x-axis
    shapeMU = np.reshape(shapeMU, [int(3), int(len(shapeMU)/3)], 'F').T
    # shapeMU = np.reshape(shapeMU, [int(len(shapeMU)/3), 3])
    shapeMU[:, 1:] = -shapeMU[:, 1:]
    shapeMU = shapeMU.reshape((68*3, 1))

    # Mean shape + shape parameters
    shapeMU += shapePC.dot(shape_param)

    Y = (s*R.dot(X.T) + t).T.reshape((204, 1))
    Sigma = np.zeros((204, 204))
    for i in range(68):
        Sigma[3*i:3*i+3, 3*i:3*i+3] = Sig_in

    inv_Gamma_s = np.linalg.inv(Gamma_s)
    inv_Gamma_v = np.linalg.inv(Gamma_v)
    
    W = np.concatenate((expPC, shapeMU), axis=1) # 3J x K+1

    term1 = inv_Gamma_s + (1-alpha)**2*W.T @ inv_Gamma_v @ W
    term2 = -(1-alpha)*W.T@inv_Gamma_v
    term3 = -(1-alpha)*inv_Gamma_v@W
    term4 = inv_Gamma_v

    inv_Gamma = np.zeros((204+29+1, 204+29+1))
    inv_Gamma[:29+1, :29+1] = term1
    inv_Gamma[:29+1, 29+1:] = term2
    inv_Gamma[29+1:, :29+1] = term3
    inv_Gamma[29+1:, 29+1:] = term4

    Gamma = np.linalg.inv(inv_Gamma)
    A = np.zeros((204+29+1, 204+29+1))
    A[:29+1, :29+1] = inv_Gamma_s
    A[:29+1, 29+1:] = term2 * alpha
    A[29+1:, 29+1:] = alpha * inv_Gamma_v

    C = np.zeros((204, 29+1+204))
    C[:, 29+1:] = np.eye(204)
    C_bar = np.zeros((29, 29+1+204))
    C_bar[:, :29] = np.eye(29)
    

    K = P @ C.T @ np.linalg.inv(C@P@C.T + Sigma)
    #print('diff before: {}'.format((Y - C @ Gamma @ A @ v)[:20, 0]))
    #print('diff after: {}'.format((K @ (Y - C @ Gamma @ A @ v))[30:50, 0]))
    #print('K: {}'.format(K[30:50, :20]))
    Psi = (np.eye(29+1+204) - K @ C) @ P
    v_new = Gamma @ A @ v + K @ (Y - C @ Gamma @ A @ v)

    ep = C_bar @ v_new
    V = C @ v_new
    P = Gamma @ A @ Psi @ A.T @ Gamma.T + Gamma

    return ep, V, Psi, P


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
        F = s * R @ PC[3*i: 3*i+3, :] # 3 x 29
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


def get_3DMM_vertices(LMs, LM_ref, h, w, bfm, s, R, t, sp, maxiter=200):    
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
        sp, ep = robust_3DMM_given_pose(LMs.transpose(), bfm, R, s, t, sp, maxiter)
    
    fitted_angles = matrix2angle(R)
    fitted_vertices = bfm.generate_vertices(sp, ep)
    transformed_vertices = bfm.transform(fitted_vertices, s, fitted_angles, t)

    return transformed_vertices, s, R, t, sp, ep


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
        F = s * R @ PC[3 * i: 3 * i + 3, :]  # 3 x 29
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


def robust_3DMM_given_pose(Y, bfm, R, s, t, sp, maxiter=200):
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
    # sp = np.zeros((n_sp, 1), dtype = np.float32)
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
    
        # # shape
        # expression = expPC.dot(ep)
        # expression = np.reshape(expression, [int(len(expression)/3), 3]).T
        # sp = estimate_non_rigid_param_3D_robust_regu_no_weight(Y, shapeMU, shapePC, model['shapeEV'][:n_sp,:], expression, s, R, t)

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
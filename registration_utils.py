from cvxopt import normal
from cvxopt.modeling import variable, op, max, sum
import pylab
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize, fsolve
from sklearn.utils import check_array
import scipy
#from OptManiMulitBallGBB import opt_mani_mulit_ball_gbb, func_q
from  scipy.special  import psi

def proj_rot(R0):
    d        = R0.shape[0]
    U, S, V	 = np.linalg.svd(R0)
    C        = np.eye(d)
    C[-1,-1] = np.linalg.det(U.dot(V.T))
    R	       = U.dot(C).dot(V.T)
    return R

def proj_orth(R0):
    U, S, V	 = np.linalg.svd(R0)
    R	       = U.dot(V.T)
    return R

def projSO3(R0):
	M	 = R0.T.dot(R0)
	D, U	 = np.linalg.eig(M)
	s	 = np.sign(np.linalg.det(R0))
	d	 = 1./np.sqrt(D)
	d[-1]	*= s
	R	 = R0.dot(U).dot(np.diag(d)).dot(U.T)
	return R

def random_rot(D):
    # D is the dimensionality
    R0 = np.random.randn(D, D)
    R = proj_rot(R0)
#    print(np.linalg.det(R0))
#    print(np.linalg.matrix_rank(R0))
    return R

def rot(q):
    """ Convert quaternion to rotation matrix"""
    M = np.array([
                    [q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]),2*(q[1]*q[3]+q[0]*q[2])],
                    [2*(q[1]*q[2]+q[0]*q[3]),q[0]**2-q[1]**2+q[2]**2-q[3]**2,2*(q[2]*q[3]-q[0]*q[1])],
                    [2*(q[1]*q[3]-q[0]*q[2]),2*(q[2]*q[3]+q[0]*q[1]),q[0]**2-q[1]**2-q[2]**2+q[3]**2]
                ])
    return M

def quaternion_from_matrix(m):
    """ Construct quaternion from rotation matrix"""
    q=[
        0.5*math.sqrt(round(m[0,0]+m[1,1]+m[2,2]+1,13)),
        0.5*math.sqrt(round(m[0,0]-m[1,1]-m[2,2]+1,13))*np.sign(m[2,1]-m[1,2]),
        0.5*math.sqrt(round(-m[0,0]+m[1,1]-m[2,2]+1,13))*np.sign(m[0,2]-m[2,0]),
        0.5*math.sqrt(round(-m[0,0]-m[1,1]+m[2,2]+1,13))*np.sign(m[1,0]-m[0,1])]
    return q


def min_rotation_over_q(x_c,y_c,s_star,sigmainv,alpha,N,q_init,opti,reg_param):
        A       = sum([alpha[0,i]*(np.reshape(x_c[:,i],(3,1))*np.reshape(x_c[:,i],(3,1)).T) for i in range(N)])
        B       = sum([alpha[0,i]*(np.reshape(x_c[:,i],(3,1))*np.reshape(y_c[:,i],(3,1)).T) for i in range(N)])
        assert(A.shape==(3,3) and B.shape==(3,3))
        f       = lambda q: T(q,s_star,A,B,sigmainv)     + reg_param*(1.-np.dot(q,q))**2
        fp      = lambda q: Tder(q,s_star,A,B,sigmainv)  + reg_param*2*(1.-np.dot(q,q))*(-2*q)
        if opti == "SLSQP_JAC":
            cons    = ({'type': 'eq', 'fun': lambda q:  1.-np.dot(q,q)})
            res     = minimize(f, q_init, method='SLSQP',  jac=fp, constraints=cons)
        elif opti == "SLSQP":
            cons    = ({'type': 'eq', 'fun': lambda q:  1.-np.dot(q,q)})
            res     = minimize(f, q_init, method='SLSQP', options={'ftol':0.00000001,'maxiter':1000},constraints=cons)
        elif opti == "BFGS_JAC":
            res     = minimize(f, q_init, method='BFGS',jac=fp)
        elif opti == "BFGS":
            res     = minimize(f, q_init, method='BFGS')
        elif opti == "Newton-CG_JAC":
            res     = minimize(f, q_init, method='Newton-CG', jac=fp)
        elif opti == "global":
            res     = basinhopping(f, q_init)
#        else:
#            tools.print_("No optimization chosen",self.verbose)
        q = res.x
#        q = q_init - 1e-4*np.asarray(Tder(q_init,s_star,A,B,sigmainv))

        return q,A,B

def opt_q_over_manifold(x_c,y_c,s_star,sigmainv,alpha,N,q_init):

        q_init = np.asarray(q_init)
        A       = sum([alpha[0,i]*(np.reshape(x_c[:,i],(3,1))*np.reshape(x_c[:,i],(3,1)).T) for i in range(N)])
        B       = sum([alpha[0,i]*(np.reshape(x_c[:,i],(3,1))*np.reshape(y_c[:,i],(3,1)).T) for i in range(N)])
        C = (s_star,A,B,sigmainv)

        q, g, out = opt_mani_mulit_ball_gbb(
            q_init,
            func_q,
            C,
            record=0,
            mxitr=600,
            gtol=1e-8,
            xtol=1e-8,
            ftol=1e-10,
            tau=1e-3)

        return q

def _impose_f_order(X):
    """Helper Function"""
    # important to access flags instead of calling np.isfortran,
    # this catches corner cases.
    if X.flags.c_contiguous:
        return check_array(X.T, copy=False, order='F'), True
    else:
        return check_array(X, copy=False, order='F'), False

def fast_dot(A, B):
    return np.dot(A,B)
    if A.ndim ==1:
        A=np.reshape(A,(1,-1))
    if B.ndim ==1:
        B=np.reshape(B,(1,-1))

    """Compute fast dot products directly calling BLAS.
    This function calls BLAS directly while warranting Fortran contiguity.
    This helps avoiding extra copies `np.dot` would have created.
    For details see section `Linear Algebra on large Arrays`:
    http://wiki.scipy.org/PerformanceTips
    Parameters
    ----------
    A, B: instance of np.ndarray
    input matrices.
    """


    if A.dtype != B.dtype:
        raise ValueError('A and B must be of the same type.')
    if A.dtype not in (np.float32, np.float64):
        raise ValueError('Data must be single or double precision float.')

    dot = scipy.linalg.get_blas_funcs('gemm', (A, B))
    A, trans_a = _impose_f_order(A)
    B, trans_b = _impose_f_order(B)
    return dot(alpha=1.0, a=A, b=B, trans_a=trans_a, trans_b=trans_b)

def compute_scale(x,y,R,sigmainv,alpha):
    assert(x.shape==(3,len(x[0])))
    assert(y.shape==(3,len(x[0])))
    assert(R.shape==(3,3))
    assert(sigmainv.shape==(3,3))
    a = np.sqrt(np.sum(alpha*np.einsum('nj,jk,nk->n', y.T, sigmainv, y.T)))
    b = np.sqrt(np.sum(alpha*np.einsum('nj,jk,nk->n', fast_dot(R,x).T, sigmainv, fast_dot(R,x).T)))
    s_star = a/b
    return s_star

def T(q,s,A,B,sigmainv):
    R   = rot(q)
    T   = fast_dot(sigmainv,(s**2*fast_dot(fast_dot(R,A),R.T)-2*s*fast_dot(R,B)))
    t   = np.trace(T)
    return t


def  Tder(q,s,A,B,sigmainv):
    d0 = np.matrix([[q[0],q[3],-q[2]],
                    [-q[3],q[0],q[1]],
                    [q[2],-q[1],q[0]]])

    d1 = np.matrix([[q[1],q[2],q[3]],
                    [q[2],-q[1],q[0]],
                    [q[3],-q[0],-q[1]]])

    d2 = np.matrix([[-q[2],q[1],-q[0]],
                    [q[1],q[2],q[3]],
                    [q[0],q[3],-q[2]]])

    d3 = np.matrix([[-q[3],q[0],q[1]],
                    [-q[0],-q[3],q[2]],
                    [q[1],q[2],q[3]]])

    R = rot(q)

    M = 2*fast_dot(sigmainv,(s**2*fast_dot(R,A.T)-s*B.T))
    d = np.array([d0,d1,d2,d3])
    a = 2*np.trace(fast_dot(M,d0))
    b = 2*np.trace(fast_dot(M,d1))
    c = 2*np.trace(fast_dot(M,d2))
    d = 2*np.trace(fast_dot(M,d3))
    m = [a,b,c,d]
    return  m

def compute_align_im(X,Y):
    A           = np.mat(X)
    B           = np.mat(Y)
    A           = A.T
    B           = B.T
    N           = A.shape[0]
    centroid_A  = np.mean(A,axis=0)
    centroid_B  = np.mean(B,axis=0)
    AA          = A-np.tile(centroid_A,(N,1))
    BB          = B-np.tile(centroid_B,(N,1))

    # SCALE AND ROTATION ESTIMATION
    num =0
    num_=0
    den =0
    den_=0
    for i in range(len(AA)):
        aa  = np.array(AA[i].tolist()[0])
        bb  = np.array(BB[i].tolist()[0])
        den +=np.dot(aa[0:3],aa[0:3])
        den_+=np.dot(aa,aa)
        num +=np.dot(bb[0:3],bb[0:3])
        num_+=np.dot(bb,bb)
    s           =  math.sqrt(num/den)
    H           = np.dot(AA.T,BB)
    U, S, Vt    = np.linalg.svd(H)
    R           = np.dot(Vt.T,U.T)

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R       = Vt.T * U.T

    # TRANSLATION ESTIMATION
    t = -s*np.dot(R,centroid_A.T) + centroid_B.T


    return R, t, s

def center_datas(x,y,alpha,N):
    assert(x.shape==(3,N))
    assert(y.shape==(3,N))
    x_b = sum([alpha[0,i]*x[:,i] for i in range(N)])/sum(alpha[0,:]) 
    y_b = sum([alpha[0,i]*y[:,i] for i in range(N)])/sum(alpha[0,:])
    x_b = np.reshape(x_b,(3,1))
    y_b = np.reshape(y_b,(3,1))
    x_c = x - np.tile(x_b,(1,N))
    y_c = y - np.tile(y_b,(1,N))

    assert(x_c.shape==(3,N) and y_c.shape==(3,N) and x_b.shape==(3,1) and y_b.shape==(3,1))
    return x_c,y_c,x_b,y_b
    
    
def E_step(r_in, r_out, pi, Sig_in, Sig_out):
    beta = []
    for i in range(len(r_in.T)):
        in_part  = scipy.stats.multivariate_normal.pdf(r_in.T[i],[0,0,0],Sig_in)
        out_part  = scipy.stats.multivariate_normal.pdf(r_out.T[i],[0,0,0],Sig_out)
        
        beta_ = np.true_divide(pi*in_part,((pi*in_part+(1.-pi)*out_part)))
        beta.append(beta_)
        
    return beta

def E_step_GUM(r,pi,sigma,gamma):
    beta = []
    for i in range(len(r.T)):
        no  = scipy.stats.multivariate_normal.pdf(r.T[i],[0,0,0],sigma)
        beta_ = pi*no/((pi*no+(1.-pi)*gamma))
        beta.append(beta_)
    return beta

def E_step_GM2(r,pi,sigma1,sigma2):
    beta = []
    for i in range(len(r.T)):
        no1  = scipy.stats.multivariate_normal.pdf(r.T[i],[0,0,0],sigma1)
        no2  = scipy.stats.multivariate_normal.pdf(r.T[i],[0,0,0],sigma2)
        beta_ = pi*no1/((pi*no1+(1.-pi)*no2))
        beta.append(beta_)
    return beta
    
def E_Zstep_Student(r,pi,sigma,invsig,a,b):
    z = []
    for i in range(len(r.T)):
        no  = scipy.stats.multivariate_normal.pdf(r.T[i],[0,0,0],sigma)
        pst = (sp.special.gamma(a[0,i]+3./2))/(np.sqrt(np.linalg.det(sigma))*sp.special.gamma(a[0,i])*np.sqrt((2*math.pi*b[0,i])**3))*(1.+r[:,i].T.dot(invsig).dot(r[:,i])/(2.*b[0,i]))**(-(a[0,i]+3./2))
        beta_ = pi*no/((pi*no+(1.-pi)*pst))
        z.append(beta_)
    return z
    
def E_step_Student(r,mu,nu,sigma_inv,N):
    a = np.zeros((1,N))
    b = np.zeros((1,N))
    w = np.zeros((1,N))
    for i in range(N):
        a[0,i] = mu[0,i] + 3./2.
        b[0,i] = nu[0,i] + np.dot(r[:,i].reshape(1, 3),np.dot(sigma_inv,r[:,i]).reshape(3, 1))*0.5
        w[0,i] = a[0,i]/b[0,i]
    return a,b,w
        
def compute_gamma(x_c,y_c,s,R,alpha,pi,N):
    gamma   = 1
    r       = y_c-s*R.dot(x_c)
    for k in range(3):
        c1  = 1./(N*(1.-pi))*sum([(1-alpha[0,i])*r[k,i] for i in range(N)])
        c2  = 1./(N*(1.-pi))*sum([(1-alpha[0,i])*r[k,i]**2 for i in range(N)])
        tmp = (c2-c1**2)
        tmp = 1./(N*(1.-pi))*sum([(1-alpha[0,i])*(c1-r[k,i])**2 for i in range(N)])
#        assert(tmp>=0)
        gamma *= 2*math.sqrt(3*tmp)
    return 1./gamma

def invpsi(X):
    from   scipy.special    import digamma
    # Based on Paul Flacker algorithm
    L=1.
    Y=np.exp(X)

    while L > 10e-8:
        Y = Y +L*np.sign(X-digamma(Y))
        L = L/2.
    return Y

def invpsi_new(X):
    from   scipy.special    import digamma, polygamma
    # Based on Paul Flacker algorithm
    M = np.floor(X >= -2.22)
    Y=M*(np.exp(X)+0.5)+(1-M)*(-1./(X-digamma(1)))
    Ydiff = 1e5
    Maxiter = 30
    iter = 1
    
    while Ydiff > 10e-3 and iter <= Maxiter:
        Ynew = Y -(digamma(Y)-X)/(polygamma(1,Y))
        Ydiff = np.abs(Y-Ynew)/Ynew
        Y = Ynew
        iter += 1
    return Y
       
def standard_reg(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    X_ =X- np.mean(X, axis = 1, keepdims = True)
    Y_ =Y- np.mean(Y, axis = 1, keepdims = True)
    D, N = X.shape
    s = s0
    R = R0

    q = quaternion_from_matrix(R)
    errs = np.zeros((1,N))
    
    for iter in range(maxiter):
        # update Sig:
        A = Y_ - s*R.dot(X_)
        Sig = np.true_divide(A.dot(A.T),N) + 1e-6*np.eye(D)
        # update scale:
        invSig = np.linalg.inv(Sig)

        alpha = np.ones((1,N))
        s = compute_scale(X_,Y_,R,invSig,alpha)

        q,A,B           = min_rotation_over_q(X_,Y_,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = np.true_divide(q,np.linalg.norm(q)) #TODO check if q is unit
        R               = rot(q)

    t = np.mean(Y, axis = 1, keepdims = True) - s*R.dot(np.mean(X, axis = 1, keepdims = True))
    
    for i in range(N):  
        x_i = X[:,i].reshape(D,1)
        y_i = Y[:,i].reshape(D,1)
        r_i = y_i-s*R.dot(x_i)-t
        errs[0,i] = (r_i.T).dot(invSig).dot(r_i)+1e-8
    return R, s, Sig, t, errs


def robust_Ri_reg(Y, X, R0, s0, maxiter):

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    D, N = X.shape
    s = s0
    R = R0
    Rt = Y.dot(np.linalg.pinv(X))
#    Rt = np.random.randn(D,D)    
    
    q = quaternion_from_matrix(R)

    pi = 0.5
    
    A = Yc - s*R.dot(Xc)
    Sig_in = A.dot(A.T)/N + 1e-6*np.eye(D)
    
    A = Y - Rt.dot(X)
    Sig_out = A.dot(A.T)/N + 1e-6*np.eye(D)
    
    for iter in range(maxiter):

        # E-step:
        beta = E_step(Yc-s*R.dot(Xc), Y-Rt.dot(X), pi, Sig_in, Sig_out)
        beta  = np.reshape(beta,(1,-1))
        alpha = beta
        
        Xc, Yc, Xb, Yb = center_datas(X,Y,alpha,N)
        
        pi      = np.mean(alpha)
                
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= alpha[0,i]
                
            Sig_in = np.true_divide(A.dot(A.T), np.sum(alpha[0])) + 1e-6*np.eye(D)
    
            # update Sig-out:
            B = Y - Rt.dot(X)
            for i in range(N):
                B[:,i] *= (1.-alpha[0,i])
                
            Sig_out = np.true_divide(B.dot(B.T), np.sum(1.-alpha[0])) + 1e-6*np.eye(D)
                    
            # update Rt:
            Delt = np.diag(1.-alpha[0])
#            Rt = (Y.dot(Delt).dot(X.T)).dot(np.linalg.inv(X.dot(Delt).dot(X.T)+1e-6*np.eye(D)))
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)
    
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q,np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha

def robust_GUM_reg(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    D, N = X.shape
    s = s0
    R = R0 
    
    q = quaternion_from_matrix(R)

    pi = 0.8
    
    A = Yc - s*R.dot(Xc)
    
    Sig_in0 = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)
    Sig_in = Sig_in0.copy()
    
    r0          = Yc-s*R.dot(Xc)
    gamma0       = np.max(r0)-np.min(r0)
    gammvect = np.zeros((1,maxiter))
    gamma = gamma0.copy()
    
    for iter in range(maxiter):

        # E-step:
        if np.sum(np.floor(np.isnan(Sig_in))):
            Sig_in = Sig_in0.copy()
            
        beta = E_step_GUM(Yc-s*R.dot(Xc), pi, Sig_in, gamma)
        beta  = np.reshape(beta,(1,-1))
        alpha = beta
        
        Xc, Yc, Xb, Yb = center_datas(X,Y,alpha,N)
        
        pi      = (1./N)*sum(beta[0])
            
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= alpha[0,i]
                
            Sig_in = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_in /= np.sum(alpha[0])
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)
    
            gamma = compute_gamma(Xc,Yc,s,R,alpha,pi,N)
            if np.isnan(gamma):
                gamma = gamma0.copy()
                
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)
        gammvect[0,iter] = gamma

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha, gammvect


def robust_GM2_reg(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
   
    D, N = X.shape
    s = s0
    R = R0 
    
    q = quaternion_from_matrix(R)

    pi = 0.8
    
    A = Yc - s*R.dot(Xc)
    
    Sig_in0 = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)
    Sig_in = Sig_in0.copy()/2
    Sig_out = Sig_in0.copy()/2
    
    beta = 0.5*np.ones((1,N))

    
    for iter in range(maxiter):

        # E-step:
            
        beta = E_step_GM2(Yc-s*R.dot(Xc), pi, Sig_in, Sig_out)
        beta  = np.reshape(beta,(1,-1))
        alpha = beta
                
        pi      = (1./N)*sum(beta[0])
            
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= alpha[0,i]
                
            Sig_in = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_in /= np.sum(alpha[0])

            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= 1.-alpha[0,i]
                
            Sig_out = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_out /= np.sum(1.-alpha[0])
                
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)
        gammvect[0,iter] = gamma

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha, gammvect
    
    
def robust_GUM_reg2(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    D, N = X.shape
    s = s0
    R = R0 
    
    q = quaternion_from_matrix(R)

    pi = 0.8
    
    A = Yc - s*R.dot(Xc)
    
    Sig_in0 = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)
    Sig_in = Sig_in0.copy()
    
    r0          = Yc-s*R.dot(Xc)
    gamma0       = np.max(r0)-np.min(r0)
    gammvect = np.zeros((1,maxiter))
    gamma = 1. #gamma0.copy()
    
    for iter in range(maxiter):

        # E-step:
        if np.sum(np.floor(np.isnan(Sig_in))):
            Sig_in = Sig_in0.copy()
            
        beta = E_step_GUM(Yc-s*R.dot(Xc), pi, Sig_in, gamma)
        beta  = np.reshape(beta,(1,-1))
        alpha = beta
        
        Xc, Yc, Xb, Yb = center_datas(X,Y,alpha,N)
        
        pi      = (1./N)*sum(beta[0])
            
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= alpha[0,i]
                
            Sig_in = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_in /= np.sum(alpha[0])
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)
    
#            gamma = compute_gamma(Xc,Yc,s,R,alpha,pi,N)
#            if np.isnan(gamma):
#                gamma = gamma0.copy()
                
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)
        gammvect[0,iter] = gamma

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha, gammvect


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    import math
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return rx, ry, rz


def objective_Student_reg(Yc, Xc, s, R, w, Sigma):
    D, N = Xc.shape
    X_trans = s * R @ Xc
    diff = Yc - X_trans
    obj = 0
    for i in range(N):
        obj += w[0][i] * diff[:, i].T @ Sigma @ diff[:, i]
    obj += np.log(np.trace(Sigma))
    return obj/2


def robust_Student_reg(Y, X, R0, s0, t0, maxiter):

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

    obj = None
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

            Sig_in = A.dot(A.T) + 1e-5*np.eye(D)
            Sig_in /= N

            # update scale:
            invSig = np.linalg.inv(Sig_in)

            for i in range(N):
                mu[0,i] = invpsi(psi(a[0,i])-1./N*sum([math.log(b[0,j]) for j in range(N)]))

            s = compute_scale(Xc,Yc,R,invSig,w)


            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,w,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)
        # Stop iteration if converge
        obj_cur = objective_Student_reg(Yc, Xc, s, R, w, Sig_in)
        if obj is None:
            obj = obj_cur
            w_pre = w.copy()
            R_pre = R.copy()
            s_pre = s
            Sig_in_pre = Sig_in
            continue
        else:
            if  obj_cur > obj:
                w = w_pre
                R = R_pre
                s = s_pre
                Sig_in = Sig_in_pre
                break
            elif obj_cur <= obj and (obj - obj_cur)/obj < 0.0001:
                break
            else:
                obj = obj_cur
                obj = obj_cur
                w_pre = w.copy()
                R_pre = R.copy()
                s_pre = s
                Sig_in_pre = Sig_in

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))


    return R, s, Sig_in, t, w


def robust_Student_Plain(Y, X, R0, s0, t0, maxiter):

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
        
    w=1*np.ones((1,N))

        
    for iter in range(maxiter):

        # E-step:   
        
                
        # update weights:
        for i in range(N):
            x_i = Xc[:,i].reshape(D,1)
            y_i = Yc[:,i].reshape(D,1)
            rep_er = y_i-s*R.dot(x_i)
            assert(rep_er.shape == (D,1))
            w[0,i] = np.true_divide( D,(rep_er.T.dot(invSig).dot(rep_er)+1e-8))
            
        Xc, Yc, Xb, Yb = center_datas(X,Y,w,N)
                    
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= float(w[0,i])
                
            Sig_in = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_in /= N
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)
              
            
            s = compute_scale(Xc,Yc,R,invSig,w)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,w,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)


            
    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, w
    

def robust_MixtureStudent_reg(Y, X, R0, s0, t0, maxiter):

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
    a = nu.copy()
    b = mu.copy()
    
    pi = 0.8
    
    for iter in range(maxiter):

        # E-step:
        beta = E_Zstep_Student(Yc-s*R.dot(Xc),pi,Sig_in,invSig,a,b)
        beta  = np.reshape(beta,(1,-1))
        
        pi      = (1./N)*sum(beta[0])
        
        a, b, w = E_step_Student(Yc-s*R.dot(Xc), mu, nu, invSig, N)
        w  = np.reshape(w,(1,-1))
        
        alpha = beta + (1.-beta)*w
        Xc, Yc, Xb, Yb = center_datas(X,Y,alpha,N)
        
        assert(w.shape==(1,N))    
        
                    
        for in_iter in range(1):
            
            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= float(alpha[0,i])
                
            Sig_in = A.dot(A.T) + 1e-6*np.eye(D)
            Sig_in /= N
            
            # update scale:
            invSig = np.linalg.inv(Sig_in)        
            
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = q/np.linalg.norm(q) #TODO check if q is unit
            R               = rot(q)

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha, pi, w
    
    
def robust_Ri_reg_iso(Y, X, R0, s0, maxiter):

    # X, Y: D*N
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    D, N = X.shape
    s = s0
    R = R0
    Rt = Y.dot(np.linalg.pinv(X))
    Rt = np.random.randn(D,D)    
    
    q = quaternion_from_matrix(R)

    pi = 0.5
    
    A = Yc - s*R.dot(Xc)
    Sig_in = A.dot(A.T)/N + 1e-6*np.eye(D)
    
    A = Y - Rt.dot(X)
    Sig_out = np.trace(A.dot(A.T)/N + 1e-6*np.eye(D))*np.eye(D)
    
    for iter in range(maxiter):

        # E-step:
        beta = E_step(Yc-s*R.dot(Xc), Y-Rt.dot(X), pi, Sig_in, Sig_out)
        beta  = np.reshape(beta,(1,-1))
        alpha = beta
        
        Xc, Yc, Xb, Yb = center_datas(X,Y,alpha,N)
        
        pi      = (1./N)*sum(beta[0])
                
        for in_iter in range(1):

            # update Sig-in:
            A = Yc - s*R.dot(Xc)
            for i in range(N):
                A[:,i] *= alpha[0,i]
                
            Sig_in = np.true_divide(A.dot(A.T), np.sum(alpha[0])) + 1e-6*np.eye(D)
    
            # update Sig-out:
            B = Y - Rt.dot(X)
            for i in range(N):
                B[:,i] *= (1.-alpha[0,i])
                
            Sig_out = np.trace(np.true_divide(B.dot(B.T), np.sum(1.-alpha[0])) + 1e-6*np.eye(D))/D*np.eye(D)
                    
            # update Rt:
            Delt = np.diag(1.-alpha[0])
            Rt = (Y.dot(Delt).dot(X.T)).dot(np.linalg.inv(X.dot(Delt).dot(X.T)+1e-6*np.eye(D))) + 1e-6*np.eye(D)
                
            # update scale:
            invSig = np.linalg.inv(Sig_in)
    
            s = compute_scale(Xc,Yc,R,invSig,alpha)
            
    
            q,A,B           = min_rotation_over_q(Xc,Yc,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
    #        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q,np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)

    t = Yb - s*R.dot(Xb)
    t = np.reshape(t,(3,1))
            
    return R, s, Sig_in, t, alpha
    
    
def robust_reg(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    X -= np.mean(X, axis = 1, keepdims = True)
    Y -= np.mean(Y, axis = 1, keepdims = True)
    D, N = X.shape
    s = s0
    R = R0
    t = 0*t0

    errs = np.zeros(maxiter)
    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)- t.dot(np.ones((1,N)))
    lams = 1./(np.linalg.norm(O, axis=0)+1e-10)
    lams = np.reshape(lams, (1,N))

    for iter in range(maxiter):

        # update Sig:
        A = Y - s*R.dot(X)-O - t.dot(np.ones((1,N)))
        Sig = A.dot(A.T)/N + 1e-6*np.eye(D)

        # update scale:
        invSig = np.linalg.inv(Sig)

        alpha = np.ones((1,N))
        s = compute_scale(X,Y-O- t.dot(np.ones((1,N))),R,invSig,alpha)

        # update the rotation:
        q,A,B           = min_rotation_over_q(X,Y-O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = q/np.linalg.norm(q) #TODO check if q is unit
        R               = rot(q)

        # update outliers:
        for i in range(1):
            Yhh = Y + np.mean(O, axis = 1).reshape((D,1))
            O = outlier_update(Yhh - s*R.dot(X) , O, invSig, lams)
        lams = 1./(np.linalg.norm(O, axis=0)+1e-8)
        lams = np.reshape(lams, (1,N))

        # update the shift:
#        t = np.mean(Y-O, axis = 1).reshape((D,1)) - s*R.dot(np.mean(X, axis = 1).reshape((D,1)))

        A = Y - s*R.dot(X)-O
        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))

    t = np.mean(Y-O, axis = 1).reshape((D,1)) - s*R.dot(np.mean(X, axis = 1).reshape((D,1)))
    return R, s, Sig, t, lams


def robust_reg_cf(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    errs = np.zeros(maxiter)
    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    lams = np.true_divide((np.linalg.norm(O-t.dot(np.ones((1,N))), axis=0)**2+1e-8),D)
    lams = np.reshape(lams, (1,N))

    for iter in range(maxiter):

        # update Sig:
        A = Y - s*R.dot(X)-O
        Sig = np.true_divide(A.dot(A.T),N) + 1e-6*np.eye(D)

        # update scale:
        invSig = np.linalg.inv(Sig)

        alpha = np.ones((1,N))
        s = compute_scale(X,Y-O,R,invSig,alpha)

        # update the rotation:
        q,A,B           = min_rotation_over_q(X,Y-O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = q/np.linalg.norm(q) #TODO check if q is unit
        R               = rot(q)

        # update outliers:
#        O = outlier_update_cf(Y - s*R.dot(X) , t, invSig, lams)
        O = outlier_update_cf_v2(Y - s*R.dot(X) , t, Sig, lams)

        lams = np.true_divide((np.linalg.norm(O-t.dot(np.ones((1,N))), axis=0)**2+1e-8),D)
        lams = np.reshape(lams, (1,N))

        # update the shift:
        a = np.zeros((D,1))
        b = 0

        for i in range(N):

            a += np.true_divide(O[:,[i]],lams[0,i])
            b += np.true_divide(1,lams[0,i])

        t = np.true_divide(a, b)

#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))

    return R, s, Sig, t, lams

def robust_reg_EM_orig(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    q = quaternion_from_matrix(R)
    mu_O = np.zeros((D,N))
    Sig_O = np.zeros((D,D))
    lams = np.true_divide((np.linalg.norm(mu_O, axis=0)**2+1e-8),D)
    lams = 1e5*np.ones((1,N))

    A = Y - s*R.dot(X)
    Sig = np.true_divide(A.dot(A.T), N) + 1e-6*np.eye(D)
    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        # update outliers:
#        O = outlier_update_cf(Y - s*R.dot(X) , t, invSig, lams)
        mu_O, Sig_O, tr_i = outlier_update_cf_orig(Y - s*R.dot(X) -t.dot(np.ones((1,N))), invSig, lams)

        for i in range(N):
            tdiff = mu_O[:,i].reshape(-1,1)
            assert(tdiff.shape == (D,1))
            lams[0,i] = np.true_divide((np.linalg.norm(tdiff)**2 + tr_i[0,i]),D)

        alpha = np.ones((1,N))
        s = compute_scale(X,Y-mu_O-t.dot(np.ones((1,N))),R,invSig,alpha)

        # update the rotation:
        q,A,B           = min_rotation_over_q(X,Y-mu_O-t.dot(np.ones((1,N))),s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = q/np.linalg.norm(q) #TODO check if q is unit
        R               = rot(q)


        # update the shift:

        t = 1./N*np.sum(Y - s*R.dot(X) - mu_O, axis = 1, keepdims=True).reshape((D,1))

        # update Sig:
        A = Y - s*R.dot(X)-t.dot(np.ones((1,N)))-mu_O
        Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)

        # update scale:
        invSig = np.linalg.inv(Sig)
        
#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))

    return R, s, Sig, t, lams
    
def robust_reg_EM(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    errs = np.zeros(maxiter)
    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    lams = np.true_divide((np.linalg.norm(O-t.dot(np.ones((1,N))), axis=0)**2+1e-8),D)
    lams = np.reshape(lams, (1,N))

    invSig = np.eye(D)

    for iter in range(maxiter):


        # E-step:
        Sig_O, mu_O = outlier_update_EM(Y - s*R.dot(X), t, invSig, lams)

        # update Sig:
        A = Y - s*R.dot(X)-mu_O
        Sig = np.true_divide(A.dot(A.T) + 1*Sig_O, N) + 0e-6*np.eye(D)

        # update scale:
        invSig = np.linalg.inv(Sig)

        alpha = np.ones((1,N))
        s = compute_scale(X,Y-mu_O,R,invSig,alpha)

        # update the rotation:
        q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = q/np.linalg.norm(q) #TODO check if q is unit
        R               = rot(q)

        # update outlier parameters:

        lams = np.true_divide((np.linalg.norm(mu_O-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8),D)
        lams = np.reshape(lams, (1,N))

        # update the shift:
        a = np.zeros((D,1))
        b = 0

        for i in range(N):

            a += np.true_divide(mu_O[:,[i]],lams[0,i])
            b += np.true_divide(1,lams[0,i])

        t = np.true_divide(a, b)

#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))

    return R, s, Sig, t, lams


def robust_reg_EMtest(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    s = s0
    R = R0
    t = t0
    
    A = Yc - s*R.dot(Xc)
    
    Sig = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    
    ti = t.dot(np.ones((1,N)))
    assert(ti.shape == (D,N))
    lams = np.true_divide((np.linalg.norm(O-ti, axis=0))**2+np.trace(Sig),D)
    lams = np.reshape(lams, (1,N))
    lams = 1e5*np.ones((1,N))
    
    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        # update outlier parameters:

        # E-step:
#        Sig_O, mu_O, lams, Sigz = outlier_update_EMAniso(Y - s*R.dot(X), t, invSig, np.eye(D), lams)
        Sig_O, mu_O, tr_sig, t_t = outlier_update_EM(Y - s*R.dot(X), t, invSig, lams)
        for i in range(N):
            tdiff = mu_O[:,i].reshape(-1,1)-t
            assert(tdiff.shape == (D,1))
            lams[0,i] = np.true_divide((np.linalg.norm(tdiff)**2 + tr_sig[0,i]),D)
        lams = np.reshape(lams, (1,N))

        for i in range(1):
    
            alpha = np.ones((1,N))
            s = compute_scale(X,Y-mu_O,R,invSig,alpha)
    
            # update the rotation:
            q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#            q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)
    
    
            # update the shift:
            a = np.zeros((D,1))
            b = 0
    
            for i in range(N):
                a += mu_O[:,[i]]/lams[0,i]
                b += 1./lams[0,i]
    
            t = a/b

            # update Sig:
            A = Y - s*R.dot(X)-mu_O
            Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)
    
            # update scale:
            invSig = np.linalg.inv(Sig)
            
#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))
#    t = t_t
    
    return R, s, Sig, t, lams


def robust_reg_EMtestAniso(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    s = s0
    R = R0
    t = t0
    
    A = Yc - s*R.dot(Xc)
    
    Sig = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    
    ti = t.dot(np.ones((1,N)))
    assert(ti.shape == (D,N))
    lams = np.true_divide((np.linalg.norm(O-ti, axis=0))**2+np.trace(Sig),D)
    lams = np.reshape(lams, (1,N))
    lams = 1e5*np.ones((1,N))
    
    invSig = np.linalg.inv(Sig)
    invSigz = np.eye(D)
    
    for iter in range(maxiter):

        # update outlier parameters:

        # E-step:
        Sig_O, mu_O, tr_sig, Sigz, SumSig = outlier_update_EMAniso(Y - s*R.dot(X), t, invSig, invSigz, lams)
        
        for i in range(N):
            tdiff = mu_O[:,[i]].reshape(-1,1)-t
            assert(tdiff.shape == (D,1))
            lams[0,i] = np.true_divide(tdiff.T.dot(invSigz).dot(tdiff)+ tr_sig[0,i], D) #np.linalg.norm(tdiff)**2 tdiff.T.dot(invSigz).dot(tdiff)

        lams = np.reshape(lams, (1,N))
#        lams /= np.linalg.norm(lams[0],1)
        
        Sigznew = np.zeros_like(Sigz)
        
        for i in range(N):
            tdiff = mu_O[:,[i]].reshape(-1,1)-t
            assert(tdiff.shape == (D,1))
            Sigznew += (1./lams[0,i])*(tdiff.dot(tdiff.T))
        
        Sigznew /= N 
    
        Sigznew += SumSig + + 1e-6*np.eye(D)
        
        invSigz = np.linalg.inv(Sigznew)
        
        for i in range(1):
    
            alpha = np.ones((1,N))
            s = compute_scale(X,Y-mu_O,R,invSig,alpha)
    
            # update the rotation:
            q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#            q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)
    
    
            # update the shift:
            a = np.zeros((D,1))
            b = 0
    
            for i in range(N):
                a += mu_O[:,[i]]/lams[0,i]
                b += 1./lams[0,i]
    
            t = a/b

            # update Sig:
            A = Y - s*R.dot(X)-mu_O
            Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)
    
            # update scale:
            invSig = np.linalg.inv(Sig)
            
#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))
#    t = t_t
    
    return R, s, Sig, t, lams



def robust_reg_varEMComplete(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    s = s0
    R = R0
    t = t0
    a0 = 1.00
    gamm0 = 1e-3
    alpha = 1e-3
    beta = 1e0
    
    m_log_p1 = np.log(0.5)
    m_log_p2 = np.log(0.5)
        
    A = Y - s*R.dot(X)
    
    Sig = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    
    ti = t.dot(np.ones((1,N)))
    assert(ti.shape == (D,N))
    lams = np.true_divide((np.linalg.norm(O-ti, axis=0))**2+np.trace(Sig),D)
    lams = np.reshape(lams, (1,N))
    lams = 1e5*np.ones((1,N))
    ri = 0.5*np.ones((1,N))
    alpha_i = np.zeros_like(ri)
    beta_i = np.zeros_like(ri)
    m_gammi = 1e-5*np.ones((1,N))
    m_log_gammi = np.zeros_like(ri)
    gamm_b_i = np.zeros_like(ri)
    
    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        # var-E-step:
    
        # q(z_i):
        for i in range(N):
            gamm_b_i[0,i] = ri[0,i]*(1./gamm0)+(1-ri[0,i])*m_gammi[0,i]
            
        Sig_O, mu_O, tr_sig, t_t = outlier_update_EM(Y - s*R.dot(X), t, invSig, 1./gamm_b_i)
        
        # q(gamm_i):
        for i in range(N):
            alpha_i[0,i] = (1.-ri[0,i])*D/2+alpha
            beta_i[0,i] = (1.-ri[0,i])*(np.linalg.norm(mu_O[:,i]-t)**2+tr_sig[0,i])/2+beta
            m_gammi[0,i] = alpha_i[0,i]/beta_i[0,i]
            m_log_gammi[0,i] = np.log(beta_i[0,i])-sp.special.digamma(alpha_i[0,i])
            
        # q(l_i):
        g1 = 0
        g2 = 0
        for i in range(N):
            d1 = np.exp(m_log_p1)*np.exp(-D/2*np.log(gamm0))*np.exp(-1./2*(1./gamm0)*(np.linalg.norm(mu_O[:,i].reshape(-1,1)-t.reshape(-1,1))**2+tr_sig[0,i]))
            d2 = np.exp(m_log_p2)*np.exp(-D/2*m_log_gammi[0,i])*np.exp(-1./2*(m_gammi[0,i])*(np.linalg.norm(mu_O[:,i].reshape(-1,1)-t.reshape(-1,1))**2+tr_sig[0,i]))
            ri[0,i] = d1/(d1+d2)
            g1 += ri[0,i]
            g2 += ri[0,i]*(np.linalg.norm(mu_O[:,i].reshape(-1,1)-t.reshape(-1,1))**2+tr_sig[0,i])
            
        gamm0 = g2/(D*g1)
        
        # q(pi_i):
        a1 = a0+np.sum([i for i in ri])
        a2 = a0+np.sum([i for i in 1.-ri])
        
        m_log_p1 = sp.special.digamma(a1) - sp.special.digamma(a1+a2)
        m_log_p2 = sp.special.digamma(a2) - sp.special.digamma(a1+a2)
#        m_log_p1 = np.log(g1/N)
#        m_log_p2 = np.log(1.-g1/N)
        
#        print(a1/(a1+a2))
        print(gamm0, 1./m_gammi)
        # M-step:
        for i in range(1):
    
            s = compute_scale(X,Y-mu_O,R,invSig,np.ones((1,N)))
    
            # update the rotation:
            q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,np.ones((1,N)),N,q,opti="SLSQP_JAC",reg_param=0)
#            q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)
    
    
            # update the shift:
            a = np.zeros((D,1))
            b = 0
    
            for i in range(N):
                a += mu_O[:,[i]]*gamm_b_i[0,i]
                b += gamm_b_i[0,i]
    
            t = a/b

            # update Sig:
            A = Y - s*R.dot(X)-mu_O
            Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)
    
            # update scale:
            invSig = np.linalg.inv(Sig)
            

    
    return R, s, Sig, t, beta_i/(alpha_i-1)
    

def robust_reg_varEMGamma(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    Xc = X.copy()
    Yc = Y.copy()
    
    Xc -= np.mean(X, axis = 1, keepdims = True)
    Yc -= np.mean(Y, axis = 1, keepdims = True)
    
    s = s0
    R = R0
    t = t0
    alpha = 0
    beta = 0
    
        
    A = Yc - s*R.dot(Xc)
    
    Sig = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    
    ti = t.dot(np.ones((1,N)))
    assert(ti.shape == (D,N))
    lams = np.true_divide((np.linalg.norm(O-ti, axis=0))**2+np.trace(Sig),D)
    lams = np.reshape(lams, (1,N))
    lams = 1e5*np.ones((1,N))
    ri = 0.5*np.ones((1,N))
    alpha_i = np.zeros_like(ri)
    beta_i = np.zeros_like(ri)
    m_gammi = 1e-5*np.ones((1,N))
    m_log_gammi = np.zeros_like(ri)
    gamm_b_i = 1e5*np.ones((1,N))
    
    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        # var-E-step:
    
        # q(z_i):

            
        Sig_O, mu_O, tr_sig, t_t = outlier_update_EM(Y - s*R.dot(X), t, invSig, gamm_b_i)
        
        # q(gamm_i):
        mg = 0
        for i in range(N):
            alpha_i[0,i] = D/2+alpha
            tdiff = mu_O[:,i].reshape(-1,1)-t
            assert(tdiff.shape == (D,1))
            beta_i[0,i] = ((np.linalg.norm(tdiff)**2 + tr_sig[0,i]))/2+beta
            m_gammi[0,i] = np.true_divide(alpha_i[0,i],beta_i[0,i])
            gamm_b_i[0,i] = np.true_divide(beta_i[0,i], alpha_i[0,i])
            m_log_gammi[0,i] = np.log(beta_i[0,i])-sp.special.digamma(alpha_i[0,i])
            mg += m_log_gammi[0,i]
            
        mg /= N

        # update aslo alpha and beta:
#        for initer in range(1):
#            print(alpha,beta)
#            alpha = invpsi(np.log(beta)-mg)
#            beta = np.true_divide(alpha,mg)
            
        # M-step:
        for i in range(1):
    
            s = compute_scale(X,Y-mu_O,R,invSig,np.ones((1,N)))
    
            # update the rotation:
            q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,np.ones((1,N)),N,q,opti="SLSQP_JAC",reg_param=0)
#            q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)
    
    
            # update the shift:
            a = np.zeros((D,1))
            b = 0

            
            for i in range(N):
                a += mu_O[:,[i]]/gamm_b_i[0,i]
                b += 1./gamm_b_i[0,i]
    
            t = a/b

            # update Sig:
            A = Y - s*R.dot(X)-mu_O
            Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)
    
            # update scale:
            invSig = np.linalg.inv(Sig)
            

    print(alpha,beta)
    return R, s, Sig, t, gamm_b_i
    
def robust_reg_EMtest0(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    errs = np.zeros(maxiter)
    q = quaternion_from_matrix(R)
    O = Y - s*R.dot(X)
    mu_O = O
    lams = np.true_divide((np.linalg.norm(O-t.dot(np.ones((1,N))), axis=0)**2+1e-8),D)
    lams = np.reshape(lams, (1,N))

    Sig_O = np.eye(D)

    for iter in range(maxiter):



        # update Sig:
        A = Y - s*R.dot(X)-mu_O
        Sig = np.true_divide(A.dot(A.T) + Sig_O, N) + 1e-6*np.eye(D)

        # update scale:
        invSig = np.linalg.inv(Sig)

        alpha = np.ones((1,N))
        s = compute_scale(X,Y-mu_O,R,invSig,alpha)

        # update the rotation:
        q,A,B           = min_rotation_over_q(X,Y-mu_O,s,invSig,alpha,N,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

        q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
        R               = rot(q)


        # update the shift:
        a = np.zeros((D,1))
        b = 0

        for i in range(N):

            a += np.true_divide(mu_O[:,[i]],lams[0,i])
            b += np.true_divide(1,lams[0,i])

        t = np.true_divide(a, b)

        # update outlier parameters:

        # E-step:
        Sig_O, mu_O, tr_sig = outlier_update_EM(Y - s*R.dot(X), t, invSig, lams)
        lams = np.true_divide((np.linalg.norm(mu_O-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8+0*tr_sig),D)
        lams = np.reshape(lams, (1,N))

#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))

    return R, s, Sig, t, lams
    
    
def robust_reg_varEM(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0
    q = quaternion_from_matrix(R)
    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)
        
    mu_s = s*np.ones(N)
    sig2_s = 1e-5*np.ones(N)

    mu_t = t.dot(np.ones((1,N))) # Y-s*R.dot(X) #t.dot(np.ones((1,N)))
    Sig_t = 1e-5*np.eye(D).reshape(-1,1).dot(np.ones((1,N)))
    Sig_t_s = 1e-5*np.ones((D,N))

    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e-5*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))

    LAM_i = np.zeros((D+1,D+1))
    mus_i = np.zeros(D+1)

    I_d = np.eye(D)
    I_d2 = np.eye(D**2)

    scale_par = 1e-5
    
    gammas = scale_par*np.ones((1,N))
    deltas = scale_par*np.ones((1,N)) #np.true_divide((np.linalg.norm(mu_t-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8),D).reshape(1,-1) 
    etas = scale_par*np.ones((1,N))

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        print(iter)
        
        # variational E-step:

        # Scale and translations:

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            a_i = np.trace(np.kron(x_i.dot(x_i.T), invSig).dot(mu_R[:,i].dot(mu_R[:,i].T)+Sig_R[:,i].reshape(D**2,D**2)))
            LAM_i[:D,:D] = invSig+I_d/deltas[0,i]
            LAM_i[:D,D:] = invSig.dot(mu_R[:,i].reshape(D,D)).dot(x_i)
            LAM_i[D:,:D] = LAM_i[:D,D:].T
            LAM_i[D:,D:] = 1/etas[0,i]+a_i
#            print(np.linalg.cond(LAM_i))
            mus_i[:D] = (invSig.dot(y_i)+t/deltas[0,i]).T
            mus_i[-1] = y_i.T.dot(invSig).dot(mu_R[:,i].reshape(D,D)).dot(x_i)+s/etas[0,i]

            Sig_TS = np.linalg.inv(LAM_i+0e-6*np.eye(D+1))
            mu_ts_i = Sig_TS.dot(mus_i)

            mu_s[i] = mu_ts_i[-1]
            sig2_s[i] = Sig_TS[D:,D:]

            mu_t[:,[i]] = mu_ts_i[:D].reshape(-1,1)
            Sig_t[:,[i]] = Sig_TS[:D,:D].reshape(-1,1)

            Sig_t_s[:,[i]] = Sig_TS[:D,D:]
#            Am = invSig+I_d/deltas[0,i]
#            Bm = invSig.dot(mu_R[:,i].reshape(D,D)).dot(x_i)
#            Dm = 1/etas[0,i]+a_i
#            
#            invA = Sig.dot(I_d-np.linalg.inv(deltas[0,i]*I_d+Sig).dot(Sig))
#            Ah = invA-np.true_divide(invA.dot(Bm).dot(Bm.T).dot(invA), Dm+Bm.T.dot(invA).dot(Bm))
#            Bh = -np.true_divide(Ah.dot(Bm),Dm)
#            Ch = Bh.T
#            Dh = 1/Dm+((1/Dm)**2)*Bm.T.dot(Ah).dot(Bm)
#            
#            ah = invSig.dot(y_i)+t/deltas[0,i]
#            bh = y_i.T.dot(invSig).dot(mu_R[:,[i]].reshape(D,D)).dot(x_i)+s/etas[0,i]
#            
#            mu_t[:,[i]] = Ah.dot(ah)+Bh.dot(bh)
#            mu_s[i] = Ch.dot(ah)+Dh.dot(bh)
#            
#            Sig_t[:,[i]] = Ah.reshape(-1,1)
#            sig2_s[i] = Dh
            
        # Rotation:

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
#            if gammas[0,i] < 1e-6:
#                Sig_R[:,[i]] = gammas[0,i]*I_d2.reshape(-1,1)
#            else:
            Sig_R[:,[i]] = np.linalg.inv((mu_s[i]**2+sig2_s[i])*np.kron(x_i.dot(x_i.T), invSig)+I_d2/gammas[0,i]).reshape(-1,1)
#            mu_R[:,[i]] = Sig_R[:,i].reshape(D**2,D**2).dot(((x_i.dot(-Sig_t_s[:,i].T+mu_s[i]*(y_i-mu_t[:,i].reshape(-1,1)).T).dot(invSig))+R/gammas[0,i]).reshape(D**2,1))
            mu_R[:,[i]] = Sig_R[:,i].reshape(D**2,D**2).dot(((invSig.dot(-Sig_t_s[:,i].reshape(-1,1)+mu_s[i]*(y_i-mu_t[:,i].reshape(-1,1))).dot(x_i.T))+R/gammas[0,i]).reshape(D**2,1))
#        print(gammas)
        # update scale:

        si = 0
        b = 0

        for i in range(N):
#            if mu_s[i]>0:
                si += np.true_divide(mu_s[i],etas[0,i])
                b += np.true_divide(1,etas[0,i])

        s = np.true_divide(si, b)

        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0

        for i in range(N):
#            print(np.amax(mu_R[:,[i]]))
                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
                b += np.true_divide(1,gammas[0,i])
            
#        print(Sig_R)
#        print(gammas)
        
        R = proj_rot(np.true_divide(Ri, b))
        
#        q,A,B           = min_rotation_over_q(np.eye(D),np.true_divide(Ri, b),1,np.eye(D),np.ones((1,3)),3,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

#        q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
#        R               = rot(q)
        
        # update the translation:

        ti = np.zeros((D,1))
        b = 0

        for i in range(N):
            ti += np.true_divide(mu_t[:,[i]],deltas[0,i])
            b += np.true_divide(1,deltas[0,i])
            
        t = np.true_divide(ti, b)
                
        # Update parameters:
        for i in range(N):
            tr_R = np.trace(Sig_R[:,i].reshape(D**2,D**2))
            tr_t = np.trace(Sig_t[:,i].reshape(D,D))
            tr_s = sig2_s[i]
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(D**2,1)-mu_R[:,i].reshape(D**2,1))**2 + tr_R), D**2)
            deltas[0,i] = np.true_divide((np.linalg.norm(t.reshape(-1,1)-mu_t[:,i].reshape(-1,1))**2 + tr_t), D)
            etas[0,i] = (s-mu_s[i])**2 + tr_s
            
        
                # Update the parameters:

        # update Sig:
        A = np.zeros((D,D))
        B_out = np.zeros((D,D))

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
#            A += Sig_t[:,i].reshape(D,D)+2*mu_R[:,i].reshape(D,D).dot(x_i).dot(mu_s[i]*(mu_t[:,i].reshape(-1,1)-y_i).T+Sig_t_s[:,i])
            A += Sig_t[:,i].reshape(D,D)+2*(mu_s[i]*(mu_t[:,i].reshape(-1,1)-y_i.reshape(-1,1))+Sig_t_s[:,i].reshape(-1,1)).dot((x_i.T).dot(mu_R[:,i].reshape(D,D).T))
            A_i = x_i.dot(x_i.T)
            B_i = (mu_R[:,i].reshape(-1,1).dot(mu_R[:,i].reshape(-1,1).T)+Sig_R[:,i].reshape(D**2,D**2))
            B_outi = np.zeros((D,D))            
            for i1 in range(D):
                B_in = np.zeros((D,D))
                for p in range(D):
                    B_in += A_i[i1,p]*B_i[p*D:(p+1)*D,i1*D:(i1+1)*D]
                B_outi += B_in.T
            B_out += (mu_s[i]**2+sig2_s[i])*B_outi

#        print(mu_t)
        YT = Y -mu_t
        Sig = np.true_divide(YT.dot(YT.T) + A.T + B_out, N) + 1e-6*np.eye(D)

        invSig = np.linalg.inv(Sig)
        
#        print(mu_R.reshape(N,D**2))
        
    return R, s, Sig, t, gammas, deltas, etas




def robust_reg_EMRt(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0
    q = quaternion_from_matrix(R)
    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)
        

    mu_t = t.dot(np.ones((1,N))) # Y-s*R.dot(X) #t.dot(np.ones((1,N)))
    Sig_t = 1e-5*np.eye(D).reshape(-1,1).dot(np.ones((1,N)))

    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e-5*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))

    Sig_t_R = np.zeros((D**3,N))
    I_d = np.eye(D)
    I_d2 = np.eye(D**2)
    
    scale_par = 1e-5
    
    gammas = scale_par*np.ones((1,N))
    deltas = scale_par*np.ones((1,N)) #np.true_divide((np.linalg.norm(mu_t-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8),D).reshape(1,-1) 

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        print(iter)
        
        # variational E-step:

        # Rotation and translation:

        for i in range(N):
            
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)           

            Am = s**2*np.kron(x_i.dot(x_i.T), invSig)+I_d2/gammas[0,i]
            Bm = s*np.kron(x_i, invSig)
            Cm = Bm.T
            Dm = invSig+I_d/deltas[0,i]
            
            invDm = np.linalg.inv(Dm)
                        
            Ah = np.linalg.inv(Am-Bm.dot(invDm).dot(Cm))
            Bh = -Ah.dot(Bm.dot(invDm))
            Ch = Bh.T
            Dh = invDm - invDm.dot(Cm).dot(Bh)
            
            ah = (s*invSig.dot(y_i).dot(x_i.T)+ R/gammas[0,i]).reshape(-1,1)
            bh = invSig.dot(y_i)+t/deltas[0,i]
            
            mu_R[:,[i]] = Ah.dot(ah)+Bh.dot(bh)
            mu_t[:,[i]] = Ch.dot(ah)+Dh.dot(bh)
            
            Sig_R[:,[i]] = Ah.reshape(-1,1)
            Sig_t[:,[i]] = Dh.reshape(-1,1)
            Sig_t_R[:,[i]] = Ch.reshape(-1,1)


        print(gammas)
        
        # update scale:

        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0

        for i in range(N):
                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
                b += np.true_divide(1,gammas[0,i])
            
        
        R = proj_rot(np.true_divide(Ri, b))
        
#        q,A,B           = min_rotation_over_q(np.eye(D),np.true_divide(Ri, b),1,np.eye(D),np.ones((1,3)),3,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

#        q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
#        R               = rot(q)
        
        # update the translation:

        ti = np.zeros((D,1))
        b = 0

        for i in range(N):
            ti += np.true_divide(mu_t[:,[i]],deltas[0,i])
            b += np.true_divide(1,deltas[0,i])
            
        t = np.true_divide(ti, b)
        
        # Update parameters:
        for i in range(N):
            tr_R = np.trace(Sig_R[:,i].reshape(D**2,D**2))
            tr_t = np.trace(Sig_t[:,i].reshape(D,D))
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(D**2,1)-mu_R[:,i].reshape(D**2,1))**2 + tr_R), D**2)
            deltas[0,i] = np.true_divide((np.linalg.norm(t.reshape(-1,1)-mu_t[:,i].reshape(-1,1))**2 + tr_t), D)
        
        # Update the parameters:

        # update Sig:
        A = np.zeros((D,D))
        B_out = np.zeros((D,D))

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            A += Sig_t[:,i].reshape(D,D)+(y_i-mu_t[:,i].reshape(-1,1)).dot((y_i-mu_t[:,i].reshape(-1,1)).T)-2*s*mu_R[:,i].reshape(D,D).dot(x_i).dot(y_i.T)
            A_i = x_i.dot(x_i.T)
            
            B_i = (mu_R[:,i].reshape(-1,1).dot(mu_R[:,i].reshape(-1,1).T)+Sig_R[:,i].reshape(D**2,D**2))
            D_i = mu_t[:,i].reshape(-1,1).dot(mu_R[:,i].reshape(-1,1).T) + Sig_t_R[:,[i]].reshape(D,D**2)
            
            B_outi1 = np.zeros((D,D))            
            for i1 in range(D):
                B_in = np.zeros((D,D))
                for p in range(D):
                    B_in += A_i[i1,p]*B_i[p*D:(p+1)*D,i1*D:(i1+1)*D]
                B_outi1 += B_in.T
                
            B_outi2 = np.zeros((D,D))            
            B_in = np.zeros((D,D))
            for p in range(D):
                B_in += x_i[p,0]*D_i[:,p*D:(p+1)*D]
            B_outi2 += B_in.T
                
            B_out += (s**2)*B_outi1 + 2*s*B_outi2


        Sig = np.true_divide(A.T + B_out, N) + 1e-6*np.eye(D)

        invSig = np.linalg.inv(Sig)
        
        
    return R, s, Sig, t, gammas, deltas
    
    

def robust_reg_fusion(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
       
    s = s0
    R = R0
    t = t0
    
    A = Y - s*R.dot(X)-t.dot(np.ones((1,N)))
    
    Sig = (1./N)*A.dot(A.T) + 1e-6*np.eye(D)

    q = quaternion_from_matrix(R)
    

    Sig = np.eye(D)
    invSig = Sig
    
    for iter in range(maxiter):

        # update outlier parameters:


        A = Y - s*R.dot(X)-t.dot(np.ones((1,N)))
    
        lams = np.true_divide(D, (np.linalg.norm(A, axis=0))**2+1e-8)
        lams = np.reshape(lams, (1,N))
    
        for i in range(5):
    
            s = compute_scale(X,Y-t.dot(np.ones((1,N))),R,invSig,lams)
    
            # update the rotation:
            q,A,B           = min_rotation_over_q(X,Y-t.dot(np.ones((1,N))),s,invSig,lams,N,q,opti="SLSQP_JAC",reg_param=0)
#            q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)
    
            q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
            R               = rot(q)
    
    
            # update the shift:
            a = np.zeros((D,1))
            b = 0
    
            for i in range(N):
                x_i = X[:,i].reshape(D,1)
                y_i = Y[:,i].reshape(D,1)
                a += lams[0,i]*(y_i-s*R.dot(x_i))
                b += lams[0,i]
    
            t = a/b

            # update Sig:
#            A = Y - s*R.dot(X)-t.dot(np.ones((1,N)))
#            Sig = np.true_divide(A.dot(A.T), N) + 1e-6*np.eye(D)
    
            # update scale:
#            invSig = np.linalg.inv(Sig)
            
#        A = Y - s*R.dot(X)-O
#        errs[iter] = np.matrix.trace(A.T.dot(invSig).dot(A))+np.log(np.linalg.det(Sig))
#    t = t_t
    
    return R, s, Sig, t, lams

    
########################################################################
def robust_reg_varEMRt(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0
    q = quaternion_from_matrix(R)
    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)

    mu_t = t.dot(np.ones((1,N))) # Y-s*R.dot(X) #t.dot(np.ones((1,N)))
    Sig_t = 1e-10*np.eye(D).reshape(-1,1).dot(np.ones((1,N)))

    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e-10*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))

    I_d = np.eye(D)
    I_d2 = np.eye(D**2)

    scale_par = 1e-10
    
    gammas = scale_par*np.ones((1,N))
    deltas = scale_par*np.ones((1,N)) #np.true_divide((np.linalg.norm(mu_t-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8),D).reshape(1,-1) 

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        print(iter)
        
        # variational E-step:

                # Update the parameters:

        # update Sig:
        A = np.zeros((D,D))
        B_out = np.zeros((D,D))

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
#            A += Sig_t[:,i].reshape(D,D)+2*mu_R[:,i].reshape(D,D).dot(x_i).dot(mu_s[i]*(mu_t[:,i].reshape(-1,1)-y_i).T+Sig_t_s[:,i])
            A += Sig_t[:,i].reshape(D,D)+2*(s*(mu_t[:,i].reshape(-1,1)-y_i.reshape(-1,1))).dot((x_i.T).dot(mu_R[:,i].reshape(D,D).T))
            A_i = x_i.dot(x_i.T)
            B_i = (mu_R[:,i].reshape(-1,1).dot(mu_R[:,i].reshape(-1,1).T)+Sig_R[:,i].reshape(D**2,D**2))
            B_outi = np.zeros((D,D))            
            for i1 in range(D):
                B_in = np.zeros((D,D))
                for p in range(D):
                    B_in += A_i[i1,p]*B_i[p*D:(p+1)*D,i1*D:(i1+1)*D]
                B_outi += B_in.T
            B_out += (s**2)*B_outi

#        print(mu_t)
        YT = Y -mu_t
        Sig = np.true_divide(YT.dot(YT.T) + A.T + B_out, N) + 1e-6*np.eye(D)

        invSig = np.linalg.inv(Sig)
        

            
        # Scale and translations:

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)

            Sig_t[:,[i]] = np.linalg.inv(invSig+I_d/deltas[0,i]).reshape(-1,1)
            mu_t[:,[i]] = Sig_t[:,[i]].reshape(D,D).dot(invSig.dot(y_i-s*mu_R[:,i].reshape(D,D).dot(x_i))+t/deltas[0,i])



        # update scale:


#        s = np.true_divide(si, b)

        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0

        for i in range(N):
#            print(np.amax(mu_R[:,[i]]))
                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
                b += np.true_divide(1,gammas[0,i])
            
#        print(Sig_R)
#        print(gammas)
        
        R = proj_rot(np.true_divide(Ri, b))
        
#        q,A,B           = min_rotation_over_q(np.eye(D),np.true_divide(Ri, b),1,np.eye(D),np.ones((1,3)),3,q,opti="SLSQP_JAC",reg_param=0)
#        q = opt_q_over_manifold(X,Y,s,invSig,alpha,N,q)

#        q               = np.true_divide(q, np.linalg.norm(q)) #TODO check if q is unit
#        R               = rot(q)
        
        # update the translation:

        ti = np.zeros((D,1))
        b = 0

        for i in range(N):
            ti += np.true_divide(mu_t[:,[i]],deltas[0,i])
            b += np.true_divide(1,deltas[0,i])
            
        t = np.true_divide(ti, b)
                
        # Update parameters:
        for i in range(N):
            tr_R = np.trace(Sig_R[:,i].reshape(D**2,D**2))
            tr_t = np.trace(Sig_t[:,i].reshape(D,D))
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(D**2,1)-mu_R[:,i].reshape(D**2,1))**2 + tr_R), D**2)
            deltas[0,i] = np.true_divide((np.linalg.norm(t.reshape(-1,1)-mu_t[:,i].reshape(-1,1))**2 + tr_t), D)
            
        
                        # Rotation:

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            if gammas[0,i] < 1e-3:
                Sig_R[:,[i]] = gammas[0,i]*I_d2.reshape(-1,1)
            else:
                Sig_R[:,[i]] = np.linalg.inv((s**2)*np.kron(x_i.dot(x_i.T), invSig)+I_d2/gammas[0,i]).reshape(-1,1)
#            mu_R[:,[i]] = Sig_R[:,i].reshape(D**2,D**2).dot(((x_i.dot(-Sig_t_s[:,i].T+mu_s[i]*(y_i-mu_t[:,i].reshape(-1,1)).T).dot(invSig))+R/gammas[0,i]).reshape(D**2,1))
            mu_R[:,[i]] = Sig_R[:,[i]].reshape(D**2,D**2).dot(((invSig.dot(s*(y_i-mu_t[:,i].reshape(-1,1))).dot(x_i.T))+R/gammas[0,i]).reshape(D**2,1))
#        print(gammas)
#        print(mu_R.reshape(N,D**2))
        
    return R, s, Sig, t, gammas, deltas
    
    

def robust_reg_varEMR(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)
        
    q = quaternion_from_matrix(R)


    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e-5*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))


    I_d = np.eye(D)

    gammas = 1e5*np.ones((1,N))

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

#        print(gammas)

        # variational E-step:
            
        # Rotation:

        for i in range(N):
            x_i = X[:,i].reshape(-1,1)
            y_i = Y[:,i].reshape(-1,1)
#            if gammas[0,i] <= 1e-3:
#            Sig_R[:,[i]] = np.linalg.inv((s**2)*np.kron(x_i.dot(x_i.T), invSig)+I_d2/gammas[0,i]).reshape(-1,1)
#            mu_R[:,[i]] = Sig_R[:,[i]].reshape(D**2,D**2).dot(((x_i.dot(s*(y_i.reshape(-1,1)-t.reshape(-1,1)).T).dot(invSig)).T+R/gammas[0,i]).reshape(D**2,1))
#            mu_R[:,[i]] = np.linalg.inv(np.kron(I_d, Sig/gammas[0,i])+np.kron(x_i.dot(x_i.T),I_d)).dot((y_i.dot(x_i.T)+Sig.dot(R)/gammas[0,i]).reshape(-1,1))
#            mu_R[:,[i]] = sp.linalg.solve_sylvester(Sig/gammas[0,i],x_i.dot(x_i.T), y_i.dot(x_i.T)+Sig.dot(R)/gammas[0,i]).reshape(-1,1)
            mu_R[:,[i]] = (gammas[0,i]*y_i.dot(x_i.T)/(x_i.T.dot(x_i))+R).reshape(-1,1)
#            else:
#                mu_R[:,[i]] = (y_i.dot(x_i.T)/(x_i.T.dot(x_i))).reshape(D**2,1)\
#                            + (1/(1+gammas[0,i]))*((np.eye(D)-(x_i.dot(x_i.T)/(x_i.T.dot(x_i)))).dot(R)).reshape(D**2,1)
#                Sig_R[:,[i]] = ((mu_R[:,[i]]-R.reshape(-1,1)).dot((mu_R[:,[i]]-R.reshape(-1,1)).T)).reshape(-1,1)
                
        # Update the parameters:

        # update Sig:
#        A = np.zeros((D,D))
#        B_out = np.zeros((D,D))
#
#        for i in range(N):
#            x_i = X[:,i].reshape(D,1)
#            y_i = Y[:,i].reshape(D,1)
#            
#            if gammas[0,i] <= 1e-3:
#                A += (y_i-t).dot((y_i-t).T) - 2*s*mu_R[:,i].reshape(D,D).dot(x_i).dot((y_i-t).T)
#                A_i = x_i.dot(x_i.T)
#                B_i = s**2*Sig_R[:,i].reshape(D**2,D**2)
#                B_outi = np.zeros((D,D))            
#                for i1 in range(D):
#                    B_in = np.zeros((D,D))
#                    for p in range(D):
#                        B_in += A_i[i1,p]*B_i[p*D:(p+1)*D,i1*D:(i1+1)*D]
#                    B_outi += B_in.T
#                B_out += B_outi

#        Sig = np.true_divide(A + B_out, N) + 1e-6*np.eye(D)

#        invSig = np.linalg.inv(Sig)
        
            
#        print(gammas)
        # update scale:

#        s = np.true_divide(si, b)


        
        # update the translation:

            
#        t = np.zeros((D,1))
#        for i in range(N):
#            x_i = X[:,i].reshape(D,1)
#            y_i = Y[:,i].reshape(D,1)
#            t += y_i-s*mu_R[:,i].reshape(D,D).dot(x_i)
#            
#        t = np.true_divide(t, N)
        
                # Update parameters:
        for i in range(N):
            tr_R = np.trace(Sig_R[:,i].reshape(D**2,D**2))
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(-1,1)-mu_R[:,i].reshape(-1,1))**2 + 1e-8+0*tr_R), D**2)
            
        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0
#        print(mu_R)
        invgammas = 1./gammas
        invgammas /= np.sum(invgammas)
        print(invgammas)
        for i in range(N):
#                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
            Ri += invgammas[0,i]*mu_R[:,[i]].reshape(D,D)
#                b += np.true_divide(1,gammas[0,i])

#        print(b)
#        R = proj_rot(np.true_divide(Ri, b))
        R = proj_rot(Ri)
                

        
    return R, s, Sig, t, gammas

def robust_reg_varEMst(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)
        
    q = quaternion_from_matrix(R)

    mu_s = s*np.ones(N)
    sig2_s = 1e-5*np.ones(N)

    mu_t =  Y-s*R.dot(X) #t.dot(np.ones((1,N)))
    Sig_t = 1e-5*np.eye(D).reshape(-1,1).dot(np.ones((1,N)))
    Sig_t_s = np.ones((D,N))

    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e-5*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))

    LAM_i = np.zeros((D+1,D+1))
    mus_i = np.zeros(D+1)

    I_d = np.eye(D)
    I_d2 = np.eye(D**2)

    scale_par = 1e-10

    gammas = scale_par*np.ones((1,N))
    deltas = scale_par*np.ones((1,N)) #np.true_divide((np.linalg.norm(mu_t-t.dot(np.ones((1,N))), axis=0)**2 + 1e-8),D).reshape(1,-1) 
    etas = scale_par*np.ones((1,N))

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

        print(iter)

                # Update the parameters:

        # update Sig:
        A = np.zeros((D,D))
        B_out = np.zeros((D,D))

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            A += Sig_t[:,i].reshape(D,D)+2*mu_R[:,i].reshape(D,D).dot(x_i).dot(mu_s[i]*(mu_t[:,i].reshape(-1,1)-y_i).T+Sig_t_s[:,i])
#            A += Sig_t[:,i].reshape(D,D)+2*(mu_s[i]*(mu_t[:,i].reshape(-1,1)-y_i.reshape(-1,1))+Sig_t_s[:,i].reshape(-1,1)).dot((x_i.T).dot(mu_R[:,i].reshape(D,D).T))
            A_i = x_i.dot(x_i.T)
            B_i = (mu_R[:,i].reshape(-1,1).dot(mu_R[:,i].reshape(-1,1).T)+Sig_R[:,i].reshape(D**2,D**2))
            B_outi = np.zeros((D,D))            
            for i1 in range(D):
                B_in = np.zeros((D,D))
                for p in range(D):
                    B_in += A_i[i1,p]*B_i[p*D:(p+1)*D,i1*D:(i1+1)*D]
                B_outi += B_in.T
            B_out += (mu_s[i]**2+sig2_s[i])*B_outi

#        print(mu_t)
        YT = Y -mu_t
        Sig = np.true_divide(YT.dot(YT.T) + A.T + B_out, N) + 1e-6*np.eye(D)

        invSig = np.linalg.inv(Sig)
        
        # variational E-step:

        # Scale and translations:

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            a_i = x_i.T.dot(R.T).dot(invSig).dot(R).dot(x_i)
            LAM_i[:D,:D] = invSig+I_d/deltas[0,i]
            LAM_i[:D,D:] = invSig.dot(R).dot(x_i)
            LAM_i[D:,:D] = LAM_i[:D,D:].T
            LAM_i[D:,D:] = 1/etas[0,i]+a_i
#            print(np.linalg.cond(LAM_i))
            mus_i[:D] = (invSig.dot(y_i)+t/deltas[0,i]).T
            mus_i[-1] = y_i.T.dot(invSig).dot(R).dot(x_i)+s/etas[0,i]

            Sig_TS = np.linalg.inv(LAM_i+1e-6*np.eye(D+1))
            mu_ts_i = Sig_TS.dot(mus_i)

            mu_s[i] = mu_ts_i[-1]
            sig2_s[i] = Sig_TS[D:,D:]

            mu_t[:,[i]] = mu_ts_i[:D].reshape(-1,1)
            Sig_t[:,[i]] = Sig_TS[:D,:D].reshape(-1,1)

            Sig_t_s[:,[i]] = Sig_TS[:D,D:]
#            Am = invSig+I_d/deltas[0,i]
#            Bm = invSig.dot(mu_R[:,i].reshape(D,D)).dot(x_i)
#            Dm = 1/etas[0,i]+a_i
#            
#            invA = Sig.dot(I_d-np.linalg.inv(deltas[0,i]*I_d+Sig).dot(Sig))
#            Ah = invA-np.true_divide(invA.dot(Bm).dot(Bm.T).dot(invA), Dm+Bm.T.dot(invA).dot(Bm))
#            Bh = -np.true_divide(Ah.dot(Bm),Dm)
#            Ch = Bh.T
#            Dh = 1/Dm+((1/Dm)**2)*Bm.T.dot(Ah).dot(Bm)
#            
#            ah = invSig.dot(y_i)+t/deltas[0,i]
#            bh = y_i.T.dot(invSig).dot(mu_R[:,[i]].reshape(D,D)).dot(x_i)+s/etas[0,i]
#            
#            mu_t[:,[i]] = Ah.dot(ah)+Bh.dot(bh)
#            mu_s[i] = Ch.dot(ah)+Dh.dot(bh)
#            
#            Sig_t[:,[i]] = Ah.reshape(-1,1)
#            sig2_s[i] = Dh
            
        # Rotation:



#        print(gammas)
        # update scale:

        si = 0
        b = 0

        for i in range(N):
            if mu_s[i]>0:
                si += np.true_divide(mu_s[i],etas[0,i])
                b += np.true_divide(1,etas[0,i])

        s = np.true_divide(si, b)

        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0

        for i in range(N):
#            print(np.amax(mu_R[:,[i]]))
                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
                b += np.true_divide(1,gammas[0,i])
            
#        print(Sig_R)
#        print(gammas)
        
        R = proj_rot(np.true_divide(Ri, b))
        
        # update the translation:

        ti = np.zeros((D,1))
        b = 0

        for i in range(N):
            ti += np.true_divide(mu_t[:,[i]],deltas[0,i])
            b += np.true_divide(1,deltas[0,i])
            
        t = np.true_divide(ti, b)
                
        # Update parameters:
        for i in range(N):
            tr_R = np.trace(Sig_R[:,i].reshape(D**2,D**2))
            tr_t = np.trace(Sig_t[:,i].reshape(D,D))
            tr_s = sig2_s[i]
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(D**2,1)-mu_R[:,i].reshape(D**2,1))**2 + tr_R), D**2)
            deltas[0,i] = np.true_divide((np.linalg.norm(t.reshape(-1,1)-mu_t[:,i].reshape(-1,1))**2 + tr_t), D)
            etas[0,i] = (s-mu_s[i])**2 + tr_s
            
        
#        print(mu_R.reshape(N,D**2))
        
    return R, s, Sig, t, gammas, deltas, etas



def robust_reg_EMR_MAP(Y, X, R0, s0, t0, maxiter):

    # X, Y: D*N
    D, N = X.shape
    s = s0
    R = R0
    t = t0

    A = Y - s*R.dot(X) - t.dot(np.ones((1,N)))
    Sig = A.dot(A.T)/N + 1e-6*np.eye(D)
        
    q = quaternion_from_matrix(R)


    mu_R = R.reshape(-1,1).dot(np.ones((1,N)))
    Sig_R = 1e5*np.eye(D**2).reshape(-1,1).dot(np.ones((1,N)))


    I_d2 = np.eye(D**2)

    gammas = 1e5*np.ones((1,N))

    invSig = np.linalg.inv(Sig)
    
    for iter in range(maxiter):

#        print(iter)

        # Update the parameters:

        # update Sig:
        A = np.zeros((D,D))

        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            
            A += (y_i-s*mu_R[:,i].reshape(D,D).dot(x_i)-t).dot((y_i-s*mu_R[:,i].reshape(D,D).dot(x_i)-t).T)
            

        Sig = np.true_divide(A , N) + 1e-6*np.eye(D)

        invSig = np.linalg.inv(Sig)
        
        # variational E-step:
            
        # Rotation:

        for i in range(N):
            x_i = X[:,i].reshape(-1,1)
            y_i = Y[:,i].reshape(-1,1)
            print(invSig)
            f       = lambda Ri: TSig(Ri,s,x_i,y_i-t.reshape(-1,1),gammas[0,i],R,invSig)
            res     = minimize(f, mu_R[:,i].reshape(D,D), method='BFGS')
            
            mu_R[:,[i]] = res.x.reshape(D**2,1)

#        print(gammas)
        # update scale:

#        s = np.true_divide(si, b)


        
        # update the translation:

            
        t = np.zeros((D,1))
        for i in range(N):
            x_i = X[:,i].reshape(D,1)
            y_i = Y[:,i].reshape(D,1)
            t += y_i-s*mu_R[:,i].reshape(D,D).dot(x_i)
            
        t = np.true_divide(t, N)
        
        # update the rotation:

        Ri = np.zeros((D,D))
        b = 0

        for i in range(N):
                Ri += np.true_divide(mu_R[:,[i]].reshape(D,D),gammas[0,i])
                b += np.true_divide(1,gammas[0,i])

        
        R = proj_rot(np.true_divide(Ri, b))
                
        # Update parameters:
        for i in range(N):
            gammas[0,i] = np.true_divide((np.linalg.norm(R.reshape(D**2,1)-mu_R[:,i].reshape(D**2,1))**2), D**2)

        
    return R, s, Sig, t, gammas


def TSig(Ri,s,xt,yt,gamma,R,invSig):
    return (yt-s*Ri*xt).T.dot(invSig).dot((yt-s*Ri*xt))+1/gamma*np.linalg.norm(R-Ri,'fro')**2

def outlier_update(A, O, invSig, lams):
    N = A.shape[1]
    mu = 1/(np.linalg.norm(invSig,2)+1e-8)
    GradO = 2*invSig.dot(O - A)
    Oh = O - mu*GradO
#    ohn = np.linalg.norm(Oh, axis=0,keepdims=True)

    Onew = Oh
    for i in range(N):
        norm_oh = np.linalg.norm(Oh[:,i])
        Onew[:,i] = Oh[:,i] *(1./norm_oh)*np.maximum(norm_oh-lams[0,i]*mu,0)
        # solve_O(np.asarray(A[:,i]), np.asarray(O[:,i]), lams[0,i], invSig)
    return Onew

def outlier_update_cf(A, t, invSig, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))

    for i in range(N):
        Onew[:,[i]] = np.linalg.inv(invSig+1/gamms[0,i]*np.eye(D)).dot(invSig.dot(A[:,i]).reshape(-1,1)+1/gamms[0,i]*t)

    return Onew

def outlier_update_cf_orig(A, invSig, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))
    Signew = np.zeros((D,D))
    tr_i = np.zeros((1,N))
    
    for i in range(N):
        Sig_i = np.linalg.inv(invSig+1./gamms[0,i]*np.eye(D))
        Onew[:,[i]] = (Sig_i.dot(invSig.dot(A[:,i]))).reshape(-1,1)
        tr_i[0,i] = np.trace(Sig_i)
        Signew += Sig_i

    return Onew, Signew, tr_i
    
def outlier_update_cf_v2(A, t, Sig, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))

    for i in range(N):
        Onew[:,[i]] = np.linalg.inv(Sig+gamms[0,i]*np.eye(D)).dot(Sig.dot(t).reshape(-1,1)+gamms[0,i]*A[:,i].reshape(-1,1))

    return Onew

def outlier_update_EM(A, t, invSig, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))
    Signew = np.zeros((D,D))
    tr_sig = np.zeros((1,N))
    t_i = 1
    
    for i in range(N):
        Signewi = np.linalg.inv(invSig+(1./gamms[0,i])*np.eye(D))
        Onew[:,[i]] = Signewi.dot(invSig.dot(A[:,i].reshape(-1,1)).reshape(-1,1)+(1./gamms[0,i])*t)
        Signew += Signewi
        tr_sig[0,i] = np.trace(Signewi)
        
    return Signew, Onew, tr_sig, t_i


def outlier_update_EMAniso(A, t, invSig, invSigz, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))
    Signew = np.zeros((D,D))
    Sigznew = np.zeros_like(invSigz)
    tr_sig = np.zeros((1,N))
    SumSig = np.zeros((D,D))
    
    for i in range(N):
        Signewi = np.linalg.inv(invSig+(1./gamms[0,i])*invSigz)
        Onew[:,[i]] = Signewi.dot(invSig.dot(A[:,i].reshape(-1,1)).reshape(-1,1)+(1./gamms[0,i])*(invSigz.dot(t)))
        Signew += Signewi
        tdiff = Onew[:,[i]].reshape(-1,1)-t
        assert(tdiff.shape == (D,1))
        tr_sig[0,i] = np.trace(invSigz.dot(Signewi))
        SumSig += (1./gamms[0,i])*Signewi
#        lams[0,i] = np.true_divide(tdiff.T.dot(invSigz).dot(tdiff)+ np.trace(invSigz.dot(Signewi)), D)
#        Sigznew += (1./gamms[0,i])*(tdiff.T.dot(tdiff)+Signewi)
        
#    Sigznew /= N   
    SumSig /= N
    
    return Signew, Onew, tr_sig, Sigznew, SumSig
    
    
def outlier_update_EMnew(A, t, invSig, gamms):
    D, N = A.shape
    Onew = np.zeros((D,N))
    Signew = np.zeros((D,D))
    t_i = np.zeros((D,1))
    Sig_i = np.zeros((D,D))
    tr_sig = np.zeros((1,N))
    
    for i in range(N):
        Signewi = np.linalg.inv(invSig+(1./gamms[0,i])*np.eye(D))
        Sig_i0 = np.linalg.inv(np.linalg.inv(invSig)+gamms[0,i]*np.eye(D))
        Onew[:,[i]] = Signewi.dot(invSig.dot(A[:,i].reshape(-1,1)).reshape(-1,1)+(1./gamms[0,i])*t)
        Signew += Signewi
        t_i += Sig_i0.dot(A[:,i].reshape(-1,1))
        Sig_i += Sig_i0
        tr_sig[0,i] = np.trace(Signewi)
        
    t_h = np.linalg.inv(Sig_i).dot(t_i)
    return Signew, Onew, tr_sig, t_h
    
def solve_O(a, o, lam, invSig):
    f = lambda o: (a-o).T.dot(invSig).dot((a-o))+lam*np.linalg.norm(o)
    res     = minimize(f, o, method='BFGS')
    return res.x

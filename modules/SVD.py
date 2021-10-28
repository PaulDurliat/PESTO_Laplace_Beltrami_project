import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from tqdm import tqdm
import modules.MainFunctions as MF
import os, sys

def get_cr(SPEC_PATH):
    files = os.listdir(SPEC_PATH)
    spec_list = []
    with tqdm(total=len(files)) as pbar:
        for file in files:
            if file[-4:] == ".npy":
                spec = np.load(SPEC_PATH+file)   
                spec_list.append(spec)
            pbar.update(1)
    data_cr = np.asarray(spec_list)
    s = data_cr.shape
    nFiles, Lmax = s[0],int(np.sqrt(s[1])-1)
    return data_cr, nFiles, Lmax
    
def reductor_matrix(v,rank):
    D = v.shape[0]
    vRed = v.copy()
    if rank < D:
        vRed[-(D-rank):,:] = 0
    return np.conj(vRed).transpose()@vRed

def pred_svd(data_cr,v,rank):
    reductor = reductor_matrix(v,rank)
    return data_cr@reductor

def complex_to_real_matrix(data_cr):
    n,m = data_cr.shape
    real_cr = np.zeros((n,2*m))
    real_cr[:,0:2*m:2] = np.real(data_cr)
    real_cr[:,1:2*m:2] = np.imag(data_cr)
    return real_cr

def complex_to_tuple(spec):
    spec_re, spec_im = np.real(spec),np.imag(spec)
    return np.ravel(np.column_stack((spec_re,spec_im)))

def tuple_to_complex(spec):
    return spec[::2] + 1j*spec[1::2]

def error_3D(test,pred,big_RY,N=46,barycentre=None,complex_cr=False):
    n = test.shape[0]
    
    if barycentre is None:
        barycentre=[N//2,N//2,N//2]
    
    RY = MF.extract_RY(big_RY,N,barycentre)
    path = "data/Test/tmp"
    
    if test.shape[1] != pred.shape[1]:
        if not complex_cr:
            l_pred = int(np.sqrt(pred.shape[1]//2)-1)
        else:
            l_pred = int(np.sqrt(pred.shape[1])-1)
        big_RY_pred = MF.find_RY(N,l_pred,RY_PATH="data/precomputedRY/",verbose=False)
        RY_pred = MF.extract_RY(big_RY_pred,N,barycentre)
    else:
        RY_pred = RY
    
    err = np.zeros(n)
    
    for i in range(n):
        spec_test = test[i,:]
        spec_pred = pred[i,:]
        
        if not complex_cr:
            spec_test = tuple_to_complex(spec_test)
            spec_pred = tuple_to_complex(spec_pred)
            
        np.save(path,spec_test)
        shape1 = MF.recompose(path+".npy",N,barycentre,RY,verbose=False)[0]
        
        np.save(path,spec_pred)
        shape2 = MF.recompose(path+".npy",N,barycentre,RY_pred,verbose=False)[0]
    
        err[i] = MF.error(shape1,shape2)

    return np.mean(np.abs(err))

def truncate(data_cr,l):
    M,D = data_cr.shape[0], (l+1)**2
    
    if data_cr.dtype == np.complex_:
        res = data_cr[:,:D].copy()
    else:
        res = data_cr[:,:2*D].copy()
        
    return res
        
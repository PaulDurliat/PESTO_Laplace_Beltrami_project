import numpy as np
from scipy import special
import matplotlib.pyplot as plt
from numpy.linalg import norm


from modules.Loader import TXTFormLoader, RAWFormLoader
from modules.RadialDecomposer import SphericalHarmonicsDecomposer
from modules.Form import Form3D
import shutil

from tqdm import tqdm
import re
import os

pi = np.pi

def decompose(OBJ_PATH,Lmax=18,big_RY=None,verbose=True):
    """
    Compute the radial decomposition of a form
    
    @param OBJ_PATH: path to the file containing the form, Type string
    @param Lmax: maximum degree of the decomposition, Type int
    @param big_RY: matrix big_RY that must contain the object points (optionnal, can be provided to avoid its calculation), Type np.array
    @param verbose: if True display progress bar, Type bool
    
    @return cr: complex radial decomposition of the form, Type np.array(((Lmax+1)**2))
    """
                
    # Compute corresponding spectrum
    loader_selector = {
        'raw' : RAWFormLoader,
        'txt' : TXTFormLoader
    }

    file_type = OBJ_PATH.split('.')[-1]

    if file_type not in loader_selector.keys(): 
        print("WARNING : File type is not implemented (.raw or .txt)")
        print("Exiting")
        return

    ratio = 1.2
    loader = loader_selector.get(file_type)(OBJ_PATH)
    form, scales   = loader.load_form(ratio=ratio)   

    OBJECT = Form3D(form, scales)

    p_points = 0.1
    points_selection = "surface"

    # Copy of the config in checkpoint
    if points_selection == "surface":
        if (Lmax + 1)**2 >= OBJECT.size_b:
            raise ValueError("Must have (Lmax + 1)**2 < OBJECT.size_b")

    # Only useful for sphere and rand_surface points selection
    nphi   = int( np.sqrt(OBJECT.size_b * p_points) )
    ntheta = nphi

    form = OBJECT.get_array_form()
    form_boundaries = OBJECT.get_array_boundaries()

    # ************** Compute Spherical Decomposition ************** #

    SHD = SphericalHarmonicsDecomposer(OBJECT)
    #SHD.fit(Lmax=Lmax,nphi=nphi,ntheta=ntheta,points_selection=points_selection)
    SHD.set_selected_points(nphi=nphi,ntheta=ntheta,points_selection=points_selection)
    SHD.fourrier_spherical_harmonics_decomposition(Lmax=Lmax,big_RY=big_RY,verbose=verbose)
    
    return SHD.cr
    
def recompose(CR_PATH,N=100,barycentre=None,RY=None,verbose=True):
    """
    Compute the shape and border of a form given its radial decomposition
    
    @param CR_PATH: path to the file containing the complex decomposition, Type string
    @param N: size of the matrix, Type int
    @param barycentre: barycentre of the form (optionnal), Type list(3)
    @param RY: matrix RY (optionnal, can be provided to avoid its calculation), Type np.array
    @param verbose: if True display progress bar, Type bool
    
    @return shape: shape of the form (filled volume, absolute value of the real part of the reconstruction), Type np.array((N,N,N))
    @return border_re: border of the absolute value of the real part of the reconstructed shape, Type np.array((N,N,N))
    @return border_im: border of the absolute value of the imaginary part of the reconstructed shape, Type np.array((N,N,N))
    """
    
    file = np.load(CR_PATH)
    cr = np.asarray(file,dtype=np.complex_)
    
    Lmax = int(np.sqrt(cr.shape[0]))-1
    
    if barycentre==None:
        c = int(N/2)
        barycentre = [c,c,c]
       
    x, y, z    = np.arange(0,N), np.arange(0,N), np.arange(0,N) 
    rx, ry, rz = np.meshgrid(x-barycentre[0],y-barycentre[1],z-barycentre[2])
    RR         = np.sqrt(rx**2 + ry**2 + rz**2)
    """
    THETA     = np.arctan2((rx**2+ry**2)**0.5,rz).ravel()
    PHI       = np.arctan2(rx,ry).ravel()
    
    shape = np.zeros((N*N*N))
       
    RY = np.zeros((N*N*N,(Lmax+1)**2),dtype=np.complex_)
    
    for l in range(Lmax+1):
        for m in range(-l, l+1):
             RY[:,l*l+l+m] = spherical_harmonic(m,l,THETA,PHI)
    """
    if RY is None:
        RY = compute_RY(N,Lmax,saveRY=False,returnRY=True,barycentre=barycentre,verbose=verbose)
        

    func = RY.dot(cr)
            
    func = func.reshape((N,N,N))
    
    harm_re = np.real(func)
    harm_im = np.imag(func)
    border_re = np.abs(harm_re)
    border_im = np.abs(harm_im)
        
    shape = border_re > RR
    border_re  = np.argwhere(shape == 1)
    border_im  = np.argwhere((border_im > RR) == 1)
    
    return shape,border_re,border_im
    
def show_reconstructed_border(border_re,border_im):
    """
    Plot the real and imaginary part of a given form reconstruction
    
    @return border_re: border of the absolute value of the real part of the reconstructed shape, Type np.array((N,N,N))
    @return border_im: border of the absolute value of the imaginary part of the reconstructed shape, Type np.array((N,N,N))
    """
    
    fig = plt.figure(figsize = (8, 8))
    
    ax = fig.add_subplot(121, projection='3d')
    X=border_re[:,0]
    Y=border_re[:,1]
    Z=border_re[:,2]
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("real")
    
    ax = fig.add_subplot(122, projection='3d')
    X=border_im[:,0]
    Y=border_im[:,1]
    Z=border_im[:,2]
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("imaginary")
    plt.show()

def error(shape1,shape2):
    """
    Compute the error (in %) between a shape2 and a reference shape.
    
    @param shape1: reference shape, Type np.array
    @param shape2: shape to test, Type np.array
    
    @return err: error between shape2 and shape1, Type float
    """

    err=np.sum(abs(1.*shape1.ravel()-1.*shape2.ravel())) / float(np.sum(shape1.ravel()))*100
    return err

def show_shape(PATH=None,OBJ=None,reconstruction=False,N=None,RY=None,verbose=False):
    """
    Plot the shape (filled volume) of the object located at the path PATH or
    of the object reconstructed from the decomposition located
    at the path PATH. The array OBJ with the shape can also be given.
    
    @param PATH: path to the file containing the object or the decomposition, Type string
    @param OBJ: shape or decomposition, Type np.array
    @param reconstruction: the PATH leads to a decomposition or not (if not, it leads to an object), Type bool
    @param N: size of the matrix containing the reconstructed object if reconstruction = True, Type int
    @param RY: matrix RY if reconstruction = True (optionnal, can be provided to avoid its calculation), Type np.array
    @param verbose: if True display progress bar, Type bool
    """
    if PATH is not None:
        if not reconstruction:
            if PATH[-3:] == "txt":
                file = open(PATH,"r").read()
                file = re.split("\n|, | ",file)
                if file[-1]=='':
                    file=file[:-1]
                data = np.asarray(file,dtype=int)
                data = data.reshape(len(file)//3,3)
            elif PATH[-3:] == "npy":
                data = np.load(PATH)
                data = np.argwhere(data==True)
            else:
                raise ValueError("File type is neither .txt nor .npy")
        else:
            if type(N)!=int:
                raise ValueError("N must be an integer.")

            data = recompose(PATH,N,RY=RY,verbose=verbose)[1]
    elif OBJ is not None:
        if not reconstruction:
            data = OBJ
        else:
            np.save("tmp/tmp",OBJ)

            data = recompose("tmp/tmp.npy",N,RY=RY,verbose=verbose)[1]
        
    else:
        raise ValueError("Missing argument PATH or OBJ.")
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X=data[:,0]
    Y=data[:,1]
    Z=data[:,2]
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

def get_object_param(OBJ_PATH):
    """
    Get the size of the bounding box of an object (as computed by the class OBJECT).
    We assume that this box is a cube.
    
    @param OBJ_PATH: path to the file containing the object, Type string
    
    @return shape: shape (filled volume) of the object, Type np.array   
    @return N: size of the bounding box, Type int
    @return barycentre: barycentre of the object, Type list
    """
    
    loader_selector = {
        'raw' : RAWFormLoader,
        'txt' : TXTFormLoader
    }

    file_type = OBJ_PATH.split('.')[-1]

    if file_type not in loader_selector.keys(): 
        print("WARNING : File type is not implemented (.raw or .txt)")
        print("Exiting")
        return

    ratio = 1.2
    loader = loader_selector.get(file_type)(OBJ_PATH)
    form, scales   = loader.load_form(ratio=ratio)   

    OBJECT = Form3D(form, scales)
    shape = OBJECT.shape
    barycentre = OBJECT.barycenter
    form = OBJECT.form
    
    if (shape[0]!=shape[1] or shape[1]!=shape[2]):
        raise ValueError("Object bounding is not a cube.")
        
    return form,shape[0],barycentre
    
def legendre(m,l,X) :
    """
    Compute Legendre associated polynomial P_m^l(X).
    
    @param m: order (0<=m<=l), Type int
    @param l: degree (l>=0), Type int
    @param X: argument (must have |x| <= 1 for x in X), Type np.array
    
    @return P_m^l(X), Type np.array
    """
    
    if m<0 or m> l:
        raise ValueError("Must have 0 =< m =< l")
    if l<0:
        raise ValueError("Must have l>=0")
    if not (np.abs(X)<=1).prod():
        raise ValueError("Must have |x| <= 1 for x in X")
    
    return special.lpmv(m,l,X)

def spherical_harmonic(m,l,Theta,Phi):
    """
    Compute Spherical Harmonica Y_m^l(Theta, Phi).
    
    @param m: order (-l<=m<=l), Type int
    @param l: degree (l>=0), Type int
    @param Theta: colatitude, Type np.array((N,M,K))
    @param Phi: longitude, type np.array((N,M,K))
    
    @return Y_m^l(Theta, Phi), Type np.array((N,M,K))
    """
    
    if m<-l or m> l:
        raise ValueError("Must have -l =< m =< l")
    
    
    PmlTmp = legendre(abs(m),l,np.cos(Theta))
    
    # Deal with negative values of m to compute the Legendre associated polynomial
    if (m<=0):   
        acc=1
        for k in range (l+m+1,l-m+1):
            acc=acc*(k)
        prod = (-1)**m/acc
          
    else: 
        prod = 1
        
    Pml=PmlTmp*prod
    
    if (m<=0): 
        acc=1
        for k in range (l+m+1,l-m+1):
            acc=acc*(k)
        nm = acc
    else: 
        acc=1
        for k in range (l-m+1,l+m+1):
            acc=acc*(k)
        nm = 1/acc
        
    return (-1)**m*Pml * np.exp((1.j) * m * Phi) * np.sqrt(np.complex((2 * l + 1) / (4 * pi) * nm))

def construct_harmonic(m,l,N):
    """
    Construct a NxNxN matrix representing the harmonic Y_m^l
    
    @param m: order (0<=m<=l), Type int
    @param l: degree (l>=0), Type int
    @param N: size of the matrix, Type int 
    
    @return shape: shape of the harmonic (filled volume, absolute value of the real part of the reconstruction), Type np.array((N,N,N))
    @return border_re: border of the absolute value of the real part of the harmonic, Type np.array((N,N,N))
    @return border_im: border of the absolute value of the imaginary part of the harmonic, Type np.array((N,N,N))
    """
    
    c     = int(N/2)
    ind=np.arange(0,N,1)

    rx, ry, rz = np.meshgrid(ind-c,ind-c,ind-c)
    RR          = np.sqrt(rx**2 + ry**2 + rz**2)
    RR[RR==0]    =0.000001

    KK=np.tile(ind,(N,N,1))
    JJ=KK.transpose(0,2,1)
    II=KK.transpose(2,1,0)
    CC=c*np.ones((N,N,N))

    THETA=np.arccos((KK-CC)/RR)
    PHI=np.arctan2(JJ-CC,II-CC)
    
    shape = np.zeros((N,N,N))
    
    func = spherical_harmonic(m,l,THETA,PHI)
    
    harm_re = np.real(func)
    harm_im = np.imag(func)
    border_re = N*np.abs(harm_re)
    border_im = N*np.abs(harm_im)
        
    shape = border_re > RR
    border_re  = np.argwhere(shape == 1)
    border_im  = np.argwhere((border_im > RR) == 1)
    
    return shape,border_re,border_im

def show_harmonic(m,l,N=100):
    """
    Plot a NxNxN matrix representing the harmonic Y_m^l
    
    @param m: order (0<=m<=l), Type int
    @param l: degree (l>=0), Type int
    @param N: size of the matrix, Type int
    """
    
    shape,border_re,border_im = construct_harmonic(m,l,N)
    
    fig = plt.figure(figsize = (8, 8))
    
    ax = fig.add_subplot(121, projection='3d')
    X=border_re[:,0]
    Y=border_re[:,1]
    Z=border_re[:,2]
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("m = "+str(m)+" ; l = "+str(l)+", real")
    
    ax = fig.add_subplot(122, projection='3d')
    X=border_im[:,0]
    Y=border_im[:,1]
    Z=border_im[:,2]
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("m = "+str(m)+" ; l = "+str(l)+", imaginary")
    plt.show()
    
def compute_RY(M,Lmax,saveRY=True,savePath="data/precomputedRY/",returnRY=False,barycentre=None,verbose=True):
    """
    Compute the RY matrix (Y_m^l(THETA,PHI)) for a matrix of size M*M*M for all l<=Lmax
    
    @param M: size of the matrix, Type int
    @param Lmax: maximum degree, Type int
    @param saveRY: if True save RY in the folder data/precomputedRY/, Type bool
    @param savePath: folder where RY is saved if asked, Type string
    @param returnRY: if True return the matrix RY, Type bool
    @param barycentre: barycentre to translate the matrix RY if needed, Type list(3)
    @param verbose: if True display progress bar, Type bool
    
    @return RY: matrix RY (only if returnRY == True), Type np.array((M**3,(Lmax+1)**2))
    """
    
    if barycentre==None:
        cM = int(M/2)
        barycentre = [cM,cM,cM]

    x, y, z    = np.arange(0,M), np.arange(0,M), np.arange(0,M) 
    rx, ry, rz = np.meshgrid(x-barycentre[0],y-barycentre[1],z-barycentre[2])
    RR         = np.sqrt(rx**2 + ry**2 + rz**2)
    THETA     = np.arctan2((rx**2+ry**2)**0.5,rz).ravel()
    PHI       = np.arctan2(rx,ry).ravel()

    RY = np.zeros((M*M*M,(Lmax+1)**2),dtype=np.complex_)
    
    if verbose:
        for i in tqdm(range((Lmax+1)**2),desc="Computing RY"):
            l = int(np.sqrt(i))
            m = i-l**2-l
            RY[:,i] = spherical_harmonic(m,l,THETA,PHI)
    else: 
        for i in range((Lmax+1)**2):
            l = int(np.sqrt(i))
            m = i-l**2-l
            RY[:,i] = spherical_harmonic(m,l,THETA,PHI)
    
    RY = RY.reshape(M,M,M,(Lmax+1)**2)
    
    if saveRY:
        path=savePath+"RY_M_"+str(M)+"_Lmax_"+str(Lmax)+"_"
        np.save(path,RY)
   
    if returnRY:
        return RY

def find_RY(N,Lmax,RY_PATH="data/precomputedRY/",verbose=True):
    """
    Find if a RY matrix consistent with N and Lmax is saved in RY_PATH. If not compute RY with M=2*N and Lmax.
    
    @param N: size of the matrix, Type int
    @param Lmax: maximum degree, Type int
    @param RY_PATH: path to the folder where the RY matrices are saved, Type string
    @param verbose: if True display progress bar, Type bool
    
    @return RY: matrix RY loaded from a file in RY_PATH if available, else computed with M=2*N and Lmax
    """
    
    files = os.listdir(RY_PATH)
    param = []
    for file in files:
        tmp = file.split("_")
        if tmp[0]=="RY":
            param.append((int(tmp[2]),int(tmp[4])))
    param = np.asarray(param)
    param = param[np.where(param[:,1]==Lmax)[0],:]
    compute = False
    if param.shape[0] == 0:
        print("WARNING in find_RY : no corresponding Lmax in find_RY, computing RY")
        compute = True
    elif N > max(param[:,0]/2):
        print("WARNING in find_RY : N > max(M) in find_R, computing RY")
        compute = True
    if compute:
        return compute_RY(2*N,Lmax,saveRY=False,returnRY=True,verbose=verbose)
    else:
        param = np.amax(param,axis=0)
        path = RY_PATH+"RY_M_"+str(param[0])+"_Lmax_"+str(param[1])+"_.npy"
        print("Loading "+path)
        return np.load(path)

def extract_RY(big_RY,N,barycentre):
    """
    Extract the matrix RY corresponding to N and barycentre from a bigger matrix big_RY
    Note that N and the barycentre must be consistent with the dimensions of big_RY
    
    @param big_RY: big matrix RY, Type np.array
    @param N: size of the return RY matrix, Type int
    @param barycentre: barycentre to translate RY, Type list(3)
    
    @return RY: matrix RY, Type np.array
    """
    
    cM = int(big_RY.shape[0]/2)
    return big_RY[cM-barycentre[1]:cM-barycentre[1]+N,cM-barycentre[0]:cM-barycentre[0]+N,cM-barycentre[2]:cM-barycentre[2]+N,:]
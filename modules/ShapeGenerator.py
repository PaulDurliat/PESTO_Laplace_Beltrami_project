import numpy as np
import matplotlib.pyplot as plt

#Generate a random radius(theta,phi) function

def define_coefs(force,deg):
    deg=deg+1
    pA=force*np.array([-1+2*np.random.rand(1)[0] for i in range(deg)])
    return pA

def polynom(x,coefs):
    r=coefs[0]
    for k in range (len(coefs)):
        r=r+coefs[k]*np.power(x,k)
    return r

def pre_radius(theta,phi, pA,pB,pC,pD):
    theta=2*theta
    acc=0
    acc+=2  *(polynom(np.cos((theta-10*np.random.rand())*(phi-10*np.random.rand())),pA))
    acc+=3  *(polynom(np.sin(theta+phi),pB))
    acc+=0.5*(polynom(np.cos(theta-phi),pC))
    acc+=2  *(polynom(np.sin(((theta-10*np.random.rand())*phi)),pD))
    acc+=1  *(polynom(np.sin(((theta))),-pD))
    return acc

def rescale_radius(pA,pB,pC,pD):
    x = np.arange(0,1*np.pi,0.1)
    y = np.arange(-np.pi,np.pi,0.1)
    X,Y = np.meshgrid(x, y) # grid of point
    Z = pre_radius(X, Y, pA,pB,pC,pD) # evaluation of the function on the grid
    a=np.max(Z)
    b=np.min(Z)
    return a,b


def radius(theta,phi, base,variation, re_a,re_b, pA,pB,pC,pD):
    #radius is a "base" radius + a positive noise of amplitude "variation"
    return base+(pre_radius(theta,phi, pA,pB,pC,pD)-re_b)*variation/(re_a-re_b)
    
    
    
#construct a shape based on the radius function

def construct_shape(N,base,variation,pA,pB,pC,pD):
    re_a,re_b=rescale_radius(pA,pB,pC,pD)
    #construct the shape of sized N based on the function radius
    shape = np.zeros((N,N,N))
    vertices  = []
    c     = int(N/2)
    ind=np.arange(0,N,1)

    rx, ry, rz = np.meshgrid(ind-c,ind-c,ind-c)
    RR          = np.sqrt(rx**2 + ry**2 + rz**2)
    RR[RR==0]    =0.1

    KK=np.tile(ind,(N,N,1))
    JJ=KK.transpose(0,2,1)
    II=KK.transpose(2,1,0)
    CC=c*np.ones((N,N,N))

    THETA=np.arccos((KK-CC)/RR)
    PHI=np.arctan2(JJ-CC,II-CC)
    val=radius(THETA,PHI,base,variation,re_a,re_b,pA,pB,pC,pD)
    shape = val > RR
    vertices  = np.argwhere(shape == 1)
    
    return shape,vertices


def plot_f():
    #plot the function radius(theta,phi) in a 3D space. It is a surface, not a 3D shape.
    x = np.arange(0,1*np.pi,0.1)
    y = np.arange(-np.pi,np.pi,0.1)
    X,Y = np.meshgrid(x, y) # grid of point
    Z = radius(X, Y) # evaluation of the function on the grid

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                          cmap=cm.RdBu,linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def show_shape(vertices,N=0):
    #show the 3D shape
    X=vertices[:,0]
    Y=vertices[:,1]
    Z=vertices[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X,Y,Z,c=Z,marker="s",s=17,alpha=1,cmap="rainbow",edgecolor='black', linewidth=0.1)
    if N!=0:
        minmax=[0,N]
        ax.set_xlim(minmax)
        ax.set_ylim(minmax) 
        ax.set_zlim(minmax) 
    plt.show()

    

def add_cuboid(S,a,b,c,c_i=0,c_j=0,c_k=0):
    N=len(S)
    c_i+=N//2
    c_j+=N//2
    c_k+=N//2
    ind=np.arange(N)
    X=np.abs(ind-c_i)<=a//2
    Y=np.abs(ind-c_j)<=b//2
    Z=np.abs(ind-c_k)<=c//2
    II=np.tile(X,(N,N,1)).transpose(2,1,0)
    JJ=np.tile(Y,(N,N,1)).transpose(0,2,1)
    KK=np.tile(Z,(N,N,1))
    return (S+II*JJ*KK)>=1

def generate_random_cuboid(N,n,connexe=False):
    cuboids=np.zeros((N,N,N))
    a,b,c=np.minimum(5+np.random.randint(30,size=3).astype(int),40)
    cuboids=add_cuboid(cuboids,a,b,c)
    for k in range (n):
        if connexe:
            x=np.argwhere(cuboids==1)
            c_i,c_j,c_k=x[np.random.choice(len(x))]-N//2
        else:
            c_i,c_j,c_k=np.minimum(1+np.random.randint(-20,20,size=3).astype(int),30)
        a,b,c=np.minimum(1+np.random.randint(20,size=3).astype(int),40)
        cuboids=add_cuboid(cuboids,a,b,c,c_i,c_j,c_k)
    ind=np.zeros(N)
    ind[:2]=1
    ind[-2:]=1
    II=np.tile(ind,(N,N,1)).transpose(2,1,0)
    JJ=np.tile(ind,(N,N,1)).transpose(0,2,1)
    KK=np.tile(ind,(N,N,1))
    return cuboids - (II+JJ+KK)
    
#import the required packages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cv2,glob
import numpy as np

#specify the color for which histogram is to be plotted
color = 'pieces/yellow'
# whether the plot should be on full scale or zoomed
zoom = 1
# load all the files in the folder
files = glob.glob(color + '*.jpg')
files.sort()
# empty arrays for separating the channels for plotting
B = np.array([])
G = np.array([])
R = np.array([])
H = np.array([])
S = np.array([])
V = np.array([])
Y = np.array([])
Cr = np.array([])
Cb = np.array([])
LL = np.array([])
LA = np.array([])
LB = np.array([])

# Data creation
# append the values from each file to the respective channel
for fi in files[:]:
    # BGR
    im = cv2.imread(fi)
    b = im[:,:,0]
    b = b.reshape(b.shape[0]*b.shape[1])
    g = im[:,:,1]
    g = g.reshape(g.shape[0]*g.shape[1])
    r = im[:,:,2]
    r = r.reshape(r.shape[0]*r.shape[1])
    B = np.append(B,b)
    G = np.append(G,g)
    R = np.append(R,r)
    # HSV
    hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    h = h.reshape(h.shape[0]*h.shape[1])
    s = hsv[:,:,1]
    s = s.reshape(s.shape[0]*s.shape[1])
    v = hsv[:,:,2]
    v = v.reshape(v.shape[0]*v.shape[1])
    H = np.append(H,h)
    S = np.append(S,s)
    V = np.append(V,v)
    # YCrCb
    ycb = cv2.cvtColor(im,cv2.COLOR_BGR2YCrCb)
    y = ycb[:,:,0]
    y = y.reshape(y.shape[0]*y.shape[1])
    cr = ycb[:,:,1]
    cr = cr.reshape(cr.shape[0]*cr.shape[1])
    cb = ycb[:,:,2]
    cb = cb.reshape(cb.shape[0]*cb.shape[1])
    Y = np.append(Y,y)
    Cr = np.append(Cr,cr)
    Cb = np.append(Cb,cb)
    # Lab
    lab = cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
    ll = lab[:,:,0]
    ll = ll.reshape(ll.shape[0]*ll.shape[1])
    la = lab[:,:,1]
    la = la.reshape(la.shape[0]*la.shape[1])
    lb = lab[:,:,2]
    lb = lb.reshape(lb.shape[0]*lb.shape[1])
    LL = np.append(LL,ll)
    LA = np.append(LA,la)
    LB = np.append(LB,lb)
    
    
# Plotting the histogram
nbins = 10
plt.figure(figsize=[20,10])
plt.subplot(2,3,1)
plt.hist2d(B, G, bins=nbins, norm=LogNorm())
plt.xlabel('B')
plt.ylabel('G')
plt.title('RGB')
if not zoom:
    plt.xlim([0,255])
    plt.ylim([0,255])
plt.colorbar()
plt.subplot(2,3,2)
plt.hist2d(B, R, bins=nbins, norm=LogNorm())
plt.colorbar()
plt.xlabel('B')
plt.ylabel('R')
plt.title('RGB')
if not zoom:
    plt.xlim([0,255])
    plt.ylim([0,255])
plt.subplot(2,3,3)
plt.hist2d(R, G, bins=nbins, norm=LogNorm())
plt.colorbar()
plt.xlabel('R')
plt.ylabel('G')
plt.title('RGB')
if not zoom:
    plt.xlim([0,255])
    plt.ylim([0,255])

plt.subplot(2,3,4)
plt.hist2d(H, S, bins=nbins, norm=LogNorm())
plt.colorbar()
plt.xlabel('H')
plt.ylabel('S')
plt.title('HSV')
if not zoom:
    plt.xlim([0,180])
    plt.ylim([0,255])

plt.subplot(2,3,5)
plt.hist2d(Cr, Cb, bins=nbins, norm=LogNorm())
plt.colorbar()
plt.xlabel('Cr')
plt.ylabel('Cb')
plt.title('YCrCb')
if not zoom:
    plt.xlim([0,255])
    plt.ylim([0,255])

plt.subplot(2,3,6)
plt.hist2d(LA, LB, bins=nbins, norm=LogNorm())
plt.colorbar()
plt.xlabel('A')
plt.ylabel('B')
plt.title('LAB')
if not zoom:
    plt.xlim([0,255])
    plt.ylim([0,255])
    plt.savefig(color + '.png',bbox_inches='tight')
else:
    plt.savefig(color + '-zoom.png',bbox_inches='tight')

plt.show()
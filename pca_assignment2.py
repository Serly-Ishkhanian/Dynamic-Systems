
!pip install pysindy

pip install scikit-learn==1.0.1

import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import scipy.io
from numpy import savetxt
from numpy import loadtxt
from pysindy import SINDy
import pysindy

"""# Load ideal and noisy data """

x1coordinate = loadtxt('/content/xcords_fromgray_c1nonoise.csv')
y1coordinate = loadtxt('/content/ycords_fromgray_c1nonoise.csv')

x2coordinate = loadtxt('/content/xcords_fromgray_c2nonoise.csv') 
y2coordinate = loadtxt('/content/ycords_fromgray_c2nonoise.csv')

x3coordinate = loadtxt('/content/xcords_fromgray_c3nonoise.csv')
y3coordinate = loadtxt('/content/ycords_fromgray_c3nonoise.csv')

x1coordinatenoisy = loadtxt('/content/noisy_xcoordinate1.csv')
y1coordinatenoisy = loadtxt('/content/noisy_ycoordinate1.csv')

x2coordinatenoisy = loadtxt('/content/noisy2_xcoordinate.csv') 
y2coordinatenoisy = loadtxt('/content/noisy2_ycoordinate.csv')

x3coordinatenoisy = loadtxt('//content/noisy3_xcoordinate.csv')
y3coordinatenoisy = loadtxt('/content/noisy3_ycoordinate.csv')

# Function to create matrix X from coordinates 

def get_data_matrix(x1,y1,x2,y2,x3,y3):

  # Get minimum number of frames (length of vectors correspond to number of frames of the videos) 
  num_of_frames = np.min(np.array([x1.shape[0],x2.shape[0],x3.shape[0]]))


  # Create the matrix X
  X = np.vstack((x1,y1,x2[:num_of_frames],y2[:num_of_frames],x3[:num_of_frames],y3[:num_of_frames]))


  return X

# Function to compute row means and center the data 

def center_data(X):

  # Get the row means
  row_means = np.mean(X,axis=1)


  # Create the mean matrix
  Xbar = np.ones((X.shape[1],1))@np.reshape(row_means,(1,row_means.shape[0]))


  # Center data 
  Xc = X-Xbar.T


  return Xc

"""# Ideal Case"""

# Create X and center data
X = get_data_matrix(x1coordinate,y1coordinate,x2coordinate,y2coordinate,x3coordinate,y3coordinate)
Xc = center_data(X)

u,s,vt = np.linalg.svd(Xc)
v = vt.T

vt.shape

s.shape

# Visualize the temporal evolutions of the first two observations 
plt.plot(s[0]*v[:,0])
plt.plot(s[1]*v[:,1])
plt.legend(['$v_1(t)$', '$v_2(t)$'])
plt.xlabel("t")

# Truncate Data (keep until the first 150 cols of Xc)
Trunctuated_xc = Xc[:, :150]


# Compute SVD of truncated data
u_tr,s_tr,vt_tr = np.linalg.svd(Trunctuated_xc)


# Matrix V (whose cols are tne temporal evolutions)
v_tr = vt_tr.T

# Visualize the temporal evolutions of the truxated data 
plt.plot(s_tr[0]*v_tr[:,0])
plt.plot(s_tr[1]*v_tr[:,1])
plt.legend(['$v_1(t)$', '$v_2(t)$'],loc=3)
plt.xlabel("t")

# Compute energy of each of the squared singular values 
s_sq = s_tr**2
ratios_sq = np.array([i/np.sum(s_sq) for i in s_sq])

plt.scatter(range(1,7),ratios_sq)
plt.xlabel("i")
plt.ylabel("$\sigma_i^2/\sum_{i}^6 \sigma_i^2 $")
plt.show()

# Cumulative energy ratio of the first two singular values 
ratios_sq[0]+ratios_sq[1]

"""# Project the data """



"""# Apply the SINDy method using different threshold values"""

Feature_Names = ["x","y"]
Optimizer = pysindy.STLSQ(threshold = 0.01)
model = pysindy.SINDy(feature_names = Feature_Names, optimizer = Optimizer)
model.fit(Xp_trunc, t = (11/226))
model.print()

Feature_Names = ["x","y"]
Optimizer = pysindy.STLSQ(threshold = 0.1)
model = pysindy.SINDy(feature_names = Feature_Names, optimizer = Optimizer)
model.fit(Xp_trunc, t = (11/226))
model.print()

# define the differential equations obtained from the SINDy method 
from scipy.integrate import odeint
def RHS(z,t):
  a=-22.253
  b=-3.237
  c=32.032
  d=2.969
  dxdt = a +b*z[1]
  dydt = c +d*z[0]

  dzdt = [dxdt,dydt]
  return dzdt

# Intial conditions

z0 = [67.17865951,-66.79797315]

# Time points 
t = np.arange(0,11,11/226)

#solve ode 
z = odeint(RHS, z0, t)

# Visualize the results 

plt.plot(t[:150],s_tr[0]*v_tr[:150,0])
plt.plot(t,z[:,0])
plt.xlabel("t")
plt.legend(['x(t)', 'Reconstructed x(t)'],loc='best')
plt.show()


plt.plot(t[:150],s_tr[1]*v_tr[:150,1])
plt.plot(t,z[:,1])
plt.xlabel("t")
plt.legend(['y(t)', 'Reconstructed y(t)'],loc='best')
plt.show()

"""# Noisy Case

"""

# Create X and center data

Xnoisy = get_data_matrix(x1coordinatenoisy,y1coordinatenoisy,x2coordinatenoisy,y2coordinatenoisy,x3coordinatenoisy,y3coordinatenoisy)
Xc_noisy = center_data(Xnoisy)

# SVD of Xc_noisy
u_noisy, s_noisy, vt_noisy = np.linalg.svd(Xc_noisy)

# Compute energy of each squared singular value and plot it 

s_noisy_sq = s_noisy**2
noisy_sum = np.sum(s_noisy_sq)
ratios_noisy=[]
for i in range(6):
  ratios_noisy.append(s_noisy_sq[i]/noisy_sum)

# Plot 
plt.scatter(range(1,7),ratios_noisy)
plt.xlabel("i")
plt.ylabel("$\sigma_i^2/\sum_{i}^6 \sigma_i^2 $")
plt.show()

# Visualize the temporal evolutions 
plt.plot(s_noisy[0]*vt_noisy[0,:])
plt.plot(s_noisy[0]*vt_noisy[1,:])
plt.legend(['$v_1(t)$', '$v_2(t)$'],loc=4)
plt.xlabel("t")
plt.show()

# Project data 
u2_noisy = u_noisy[:,:2]
Xp_noisy = u2_noisy.T @ Xc_noisy
Xp_noisy= Xp_noisy.T

"""# Apply the SINDy method on the projected data """

Feature_Names2 = ["x","y"]
Optimizer2 = pysindy.STLSQ(threshold = 0.01)
noisy_model = pysindy.SINDy(feature_names = Feature_Names2, optimizer = Optimizer2)
noisy_model.fit(Xp_noisy, t =(12/314))
noisy_model.print()

v_noisy = vt_noisy.T

# define your function 
from scipy.integrate import odeint
def RHS(z,t):
  a=-3.285
  b=-0.475
  c=-2.406
  d= 0.181

  dxdt = a +b*z[1]
  dydt = c +d*z[0]
  dzdt = [dxdt,dydt]
  return dzdt

# Intial conditions

z0 = [-7.1921818,-23.58491566]


# Time points 
t = np.arange(0,12,12/314)

#solve ode 
z = odeint(RHS, z0, t)

plt.plot(t,s_noisy[0]*v_noisy[:,0])
plt.plot(t,z[:,0])
plt.xlabel("t")
plt.legend(['x(t)', 'Reconstructed x(t)'],loc='best')
plt.show()

plt.plot(t,s_noisy[1]*v_noisy[:,1])
plt.plot(t,z[:,1])
plt.xlabel("t")
plt.legend(['y(t)', 'Reconstructed y(t)'],loc='best')
plt.show()


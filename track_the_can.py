
import numpy as np 
import cv2 
import matplotlib.pyplot as plt 
import scipy.io
from numpy import savetxt



"""# Load data wit no noise (IDEAL CASE)"""

c1 = scipy.io.loadmat("cam1_1.mat")
c1nonoise = c1["vidFrames1_1"]

c2 = scipy.io.loadmat("cam2_1.mat")
c2nonoise = c2["vidFrames2_1"]

c3 = scipy.io.loadmat("cam3_1.mat")
c3nonoise = c3["vidFrames3_1"]

height1, width1, channels, num_of_frames = c1nonoise.shape
height2,width2, channels, number_of_frames2 = c2nonoise.shape
height3,width3, channels, number_of_frames3 = c3nonoise.shape

"""# Load data with noise (Noisy Case)"""

c12 = scipy.io.loadmat("cam1_2.mat")
c1noise = c12["vidFrames1_2"]

c22 = scipy.io.loadmat("cam2_2.mat")
c2noise = c22["vidFrames2_2"]

c32 = scipy.io.loadmat("cam3_2.mat")
c3noise = c32["vidFrames3_2"]

height12, width12, channels, number_of_frames12 = c1noise.shape
height22,width22, channels, number_of_frames22 = c2noise.shape
height32,width32, channels, number_of_frames32 = c3noise.shape

def turn_to_gray(videodata):

  number_of_frames = videodata.shape[3]

  grey_frames = [] 

  # turn to greyscale 
  for i in range(number_of_frames):
  
    grey = np.mean(videodata[:,:,:,i], axis=2)
    grey_frames.append(grey)

  grey_frames = np.array(grey_frames) # shape is (number of frames, height, width)

  return grey_frames

"""# Turn the data to gray"""

c1_gray = turn_to_gray(c1nonoise)
c2_gray = turn_to_gray(c2nonoise)
c3_gray = turn_to_gray(c3nonoise)

c1noisy_gray =turn_to_gray(c1noise)
c2noisy_gray =turn_to_gray(c2noise)
c3noisy_gray =turn_to_gray(c3noise)

"""The function takes input:
Reference can, gray video data, starth, endh, startw, endw.



Where starth, endh represents the reigon of the interest for the height (helps save computation time).


If not specifying a specific region for height(example), input 0 for starth, Xgray.shape[1] for endh.

"""

def track(my_can, Xgray, starth, endh,startw, endw):
  
  # Define the coordinate arrays 
  xcoordinate = []
  ycoordinate = [] 



  # Define height and width of the gray can 
  # It'll also be the height and width of the regions we will be taking throughout the frames
  h = my_can.shape[0]
  w = my_can.shape[1]


  for frame in range(Xgray.shape[0]):
    value = 10000000

    for wid in range((endw - startw)-w):
      for heigh in range((endh - starth)-h):


        # Take a submatrix from the frame to test if it's your can 
        test_can1 = Xgray[frame, starth + heigh : starth + h + heigh, startw + wid: startw + wid + w]

        #calculate frob norm 
        frob_norm = np.linalg.norm( my_can - test_can1 , "fro")

        if frob_norm <= value:

          value = frob_norm


          #save upper left coordinates 
          xl = startw + wid
          yl = starth + heigh
          
          # Center 
          x = (xl +xl +w)/2
          y = (yl +yl +h)/2

    xcoordinate.append(x)
    ycoordinate.append(y)


  return xcoordinate, ycoordinate

"""# XY Coordinate for C1 No Noise"""

# The can I'm using to compare 
plt.imshow(c1nonoise[235:300,320:375,:,0])
plt.show()

my_can = c1_gray[0,235:300,320:375]

# ROI
starth1 = 50 
endh1 = 435
startw1 = 300
endw1 = 389

# Track
x1coordinate, y1coordinate = track(my_can,c1_gray,starth1,endh1,startw1,endw1)

# Visualize the tracking
for i in range(c1nonoise.shape[3]):
  plt.imshow(c1nonoise[:,:,:,i])
  plt.scatter(x1coordinate[i],y1coordinate[i])
  plt.show()

# Save the coordinates
savetxt('xcords_fromgray_c1nonoise.csv', x1coordinate)
savetxt('ycords_fromgray_c1nonoise.csv', y1coordinate)

"""# C1 With noise """

startw12 = 300 
endw12 = 450 

# No specification on height 
starth12 = 0
endh12 = c1noise.shape[0]

# Use the same reference can 
my_can = c1_gray[0,235:300,320:375]


# Get track the can in noisy video
x1coordinate_noisy, y1coordinate_noisy = track(my_can,c1noisy_gray,starth12,endh12,startw12,endw12)

# Save the coordinates
savetxt('noisy_xcoordinate1.csv', x1coordinate_noisy)
savetxt('noisy_ycoordinate1.csv', y1coordinate_noisy)

# Visualize the tracking
for i in range(c1noise.shape[3]):
  plt.imshow(c1noise[:,:,:,i])
  plt.scatter(x1coordinate_noisy[i],y1coordinate_noisy[i])
  plt.show()

"""# XY Coordinate for C2 with no noise"""

# Picture of the can that's being used as a reference 
plt.imshow(c2nonoise[275:360,260:325,:,0])

my_can = c2_gray[0,275:360,260:325]

# Specify region of interest
starth2 = 60
endh2 = 425
startw2 = 240
endw2 = 360

# Track 
x2coordinate, y2coordinate = track(my_can, c2_gray,starth2,endh2,startw2,endw2 )

# Visualize the tracking
for i in range(c2nonoise.shape[3]):

  plt.imshow(c2nonoise[:,:,:,i])
  plt.scatter(x2coordinate[i],y2coordinate[i])
  plt.show()

# Save the coordinates
savetxt('xcords_fromgray_c2nonoise.csv', x2coordinate)
savetxt('ycords_fromgray_c2nonoise.csv', y2coordinate)

"""# C2 with noise """

my_can = c2_gray[0,275:360,260:325]

# Specify the region of interest
starth22 = 0
endh22 = c2nonoise.shape[0]
startw22 = 150
endw22 = 500

# Track 
noisy2_xcoordinate, noisy2_ycoordinate = track(my_can, c2noisy_gray,starth22, endh22,startw22,endw22 )

# Save the coordinates
savetxt('noisy2_xcoordinate.csv', noisy2_xcoordinate)
savetxt('noisy2_ycoordinate.csv', noisy2_ycoordinate)

# Visualize the tracking
for i in range(c2noise.shape[3]):
  plt.imshow(c2noise[:,:,:,i])
  plt.scatter(noisy2_xcoordinate[i],noisy2_ycoordinate[i])
  plt.show()

"""# XY Coordinate for C3 No noise """

# Picture of the can that I will be using to track the coordinates of the can in the frames 
# Using frame 30 since it has a clear picture 
# Also noticed that using smaller region gave more accurate results 

plt.imshow(c3nonoise[30,300:320,410:470,:,30])
plt.show()

my_can = c3_gray[30,300:320,410:470]


# Reigon of interest 
starth3 = 200
endh3 = 350
startw3 = 0
endw3 = c3nonoise.shape[1]

# Track 
x3coordinate, y3coordinate = track(my_can,c3_gray,starth3,endh3,startw3,endw3)

# Save the coordinates
savetxt('xcords_fromgray_c3nonoise.csv', x3coordinate)
savetxt('ycords_fromgray_c3nonoise.csv', y3coordinate)

# Visualize the tracking
for i in range(c3nonoise.shape[3]):
  print(i)
  plt.imshow(c3nonoise[:,:,:,i])
  plt.scatter(x3coordinate[i],y3coordinate[i])
  plt.show()

"""# Camera 3 with noise"""

my_can = c3_gray[30,300:320,410:470]

# Reigon of interest starts and ends at 
starth32 = 100
endh32 = 450
startw32 = 0
endw32 = c3noise.shape[3]

# Track 
noisy3_xcoordinate, noisy3_ycoordinate = track(my_can,c3noisy_gray,starth32,endh32,startw32,endw32)

# Save the coordinates
savetxt('noisy3_xcoordinate.csv', noisy3_xcoordinate)
savetxt('noisy3_ycoordinate.csv', noisy3_ycoordinate)

# Visualize the tracking
for i in range(c3noise.shape[3]):
  plt.imshow(c3noise[:,:,:,i])
  plt.scatter(noisy3_xcoordinate[i],noisy3_ycoordinate[i])
  plt.show()






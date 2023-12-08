import cv2
import numpy as np

#Display the image
input = cv2.imread('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/mandrill.tif')
cv2.imshow('Input', input)
cv2.waitKey(0)

#Display the image as gray image
input = cv2.imread('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/mandrill.tif', cv2.IMREAD_GRAYSCALE)
cv2.imshow('Gray_img', input)
cv2.waitKey(0)

#Convert the image to luminescence chroma blue and red
yuv_img = cv2.imread('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/mandrill.tif', cv2.IMREAD_COLOR)
y,u,v = cv2.split(yuv_img)
cv2.imshow('Y channel', y)
cv2.imshow('U channel', u)
cv2.imshow('V channel', v)
cv2.waitKey(0)

#Interpolation
input = cv2.imread('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/mandrill.tif')
img = cv2.resize(input, (450,400), interpolation = cv2.INTER_AREA)
cv2.imshow('INTER_AREA', img)
cv2.waitKey(0)

img = cv2.resize(input, None, fx = 1.2, fy = 1.2, interpolation = cv2.INTER_LINEAR)
cv2.imshow('INTER_LINEAR', img)
cv2.waitKey(0)

img = cv2.resize(input, None, fx = 1.2, fy = 1.2, interpolation = cv2.INTER_CUBIC)
cv2.imshow('INTER_CUBIC', img)
cv2.waitKey(0)

#image rotation
row, col = input.shape[0:2]
rotation_matrix = cv2.getRotationMatrix2D((row/2,col/2), 30, 0.7)
rotated_img = cv2.warpAffine(input, rotation_matrix, (row, col))
cv2.imshow('Rotated_img', rotated_img)
cv2.waitKey(0)

#Affine transformation
src_points = np.float32([[0,0], [col-1,0], [0,row-1]])
dst_points = np.float32([[0,0,], [int(0.6*(col-1)),0], [int(0.4*(col-1)), row-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
affine_img = cv2.warpAffine(input, affine_matrix, (row, col))
cv2.imshow('Affine_matrix', affine_img)
cv2.waitKey(0)

#Projective Transformation
src_points = np.float32([[0,0], [col-1,0], [0,row-1], [col-1, row-1]])
dst_points = np.float32([[0,0], [col-1, 0], [int(0.33*col), row-1], [int(0.66*col), row-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
projective_img = cv2.warpPerspective(input, projective_matrix, (row, col))
cv2.imshow('Projective_img', projective_img)
cv2.waitKey(0)

#Blurring the image
img = cv2.imread('C:/Users/ADMIN/Downloads/Data Sets_SDC/opencv_config_files/Day 5/plane.bmp')
kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
kernel_3x3 = np.ones((3,3), np.float32)/9.0
kernel_5x5 = np.ones((5,5), np.float32)/25.0
cv2.imshow('Input', input)
output = cv2.filter2D(img, -1, kernel_identity)
cv2.imshow('Identity_filter', output)
output = cv2.filter2D(img, -1, kernel_3x3)
cv2.imshow('3x3_Kernel', output)
output = cv2.filter2D(img, -1, kernel_5x5)
cv2.imshow('5x5_kernel', output)
cv2.waitKey(0)
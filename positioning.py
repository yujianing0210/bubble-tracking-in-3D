import cv2
import numpy as np

# Load camera matrix and distortion coefficients
camera_matrix = np.load('2d_checkboard/camera_matrix.npy')
dist_coeffs = np.load('2d_checkboard/dist_coeffs.npy')
print("camera_matrix:",camera_matrix)
print("dist_coeffs:", dist_coeffs)

# Load images taken from different angles
image1 = cv2.imread('images/group1.jpg') #rectified_left_image.jpg
image2 = cv2.imread('images/group2.jpg') #rectified_right_image.jpg

#-------------------------------------------------#

# Convert the image from BGR to HSV color space
hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)

# Define the lower and upper boundaries for the red and yellow color in HSV color space
# lower_red1 = np.array([0, 100, 100])
# upper_red1 = np.array([10, 255, 255])
# lower_red2 = np.array([350, 100, 100])
# upper_red2 = np.array([360, 255, 255])
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([40, 255, 255])

# Create a mask for the red color
# red_mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)
# red_mask2 = cv2.inRange(hsv1, lower_red2, upper_red2)
# red_mask3 = cv2.inRange(hsv2, lower_red1, upper_red1)
# red_mask4 = cv2.inRange(hsv2, lower_red2, upper_red2)
yellow_mask1 = cv2.inRange(hsv1, lower_yellow, upper_yellow)
yellow_mask2 = cv2.inRange(hsv2, lower_yellow, upper_yellow)
# mask = cv2.bitwise_or(mask1, mask2)

# 使用颜色掩码提取红色和黄色圆点的像素位置
# red_points_img1 = cv2.findNonZero(red_mask1)
# red_points_img11 = cv2.findNonZero(red_mask2)
# red_points_img2 = cv2.findNonZero(red_mask3)
# red_points_img22 = cv2.findNonZero(red_mask4)
yellow_points_img1 = cv2.findNonZero(yellow_mask1)
yellow_points_img2 = cv2.findNonZero(yellow_mask2)
print("yellow_points_img1:", yellow_points_img1)
print("yellow_points_img2:", yellow_points_img2)

# 进行特征提取和匹配
# 在这个示例中，使用ORB特征提取器和匹配器
orb = cv2.ORB_create()

# 检测特征点并计算描述子
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

# 使用暴力匹配器进行特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# 对匹配结果按照特征点距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 选择前N个最佳匹配
N = 100
good_matches = matches[:N]

# 提取匹配的特征点对应的图像坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 进行三角测量
# 将图像坐标转换为相机坐标
points1 = cv2.undistortPoints(src_pts, camera_matrix, dist_coeffs)
points2 = cv2.undistortPoints(dst_pts, camera_matrix, dist_coeffs)

# 进行三角测量
_, R, t, _ = cv2.solvePnPRansac(points1, points2, camera_matrix, dist_coeffs)

# 将旋转向量转换为旋转矩阵
R_matrix, _ = cv2.Rodrigues(R)

# Calculate the 3D coordinates of the red dots in camera coordinate system
# red_points_cam1 = cv2.perspectiveTransform(red_points_img1, np.hstack((R_matrix, t)))
# red_points_cam11 = cv2.perspectiveTransform(red_points_img11, np.hstack((R_matrix, t)))
# ed_points_cam2 = cv2.perspectiveTransform(red_points_img2, np.hstack((R_matrix, t)))
# red_points_cam22 = cv2.perspectiveTransform(red_points_img22, np.hstack((R_matrix, t)))

# Calculate the 3D coordinates of the yellow dots in camera coordinate system
yellow_points_cam1 = cv2.perspectiveTransform(yellow_points_img1, np.hstack((R_matrix, t)))
yellow_points_cam2 = cv2.perspectiveTransform(yellow_points_img2, np.hstack((R_matrix, t)))

# Print the 3D coordinates of the red dots
# print("Red Dot 1 in Camera Coordinate System:")
# for point in red_points_cam1:
#     print(point[0])
# print("Red Dot 2 in Camera Coordinate System:")
# for point in red_points_cam2:
#     print(point[0])

# Print the 3D coordinates of the yellow dots
print("Yellow Dot 1 in Camera Coordinate System:")
for point in yellow_points_cam1:
    print(point[0])

print("Yellow Dot 2 in Camera Coordinate System:")
for point in yellow_points_cam2:
    print(point[0])

import cv2
import numpy as np

# 准备标定板的角点坐标
pattern_size = (8, 8)
square_size = 1  # cm

# 存储角点坐标的列表
object_points = []
image_points = []

# 逐一读取图像并查找角点
for i in range(10):
    image_path = f'3d_checkerboard/3d-{i}.jpg'
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点 - XY平面
    found_xy, corners_xy = cv2.findChessboardCorners(gray, pattern_size, None)

    # 查找棋盘格角点 - XZ平面
    found_xz, corners_xz = cv2.findChessboardCorners(gray, pattern_size, None, cv2.CALIB_CB_EXHAUSTIVE)

    # 查找棋盘格角点 - YZ平面
    found_yz, corners_yz = cv2.findChessboardCorners(gray, pattern_size[::-1], None, cv2.CALIB_CB_EXHAUSTIVE)

    if found_xy and found_xz and found_yz:
        # 添加世界坐标系中的角点坐标
        object_points.append(np.concatenate((corners_xy, corners_xz, corners_yz)))

        # 添加图像坐标系中的角点坐标
        image_points.append(np.concatenate((corners_xy, corners_xz, corners_yz)))

# 进行相机校准
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 打印相机参数
print("camera matrix:")
print(camera_matrix)
print("distortion coefficients:")
print(dist_coeffs)

# 保存相机矩阵和畸变系数为.npy文件
np.save('3d/camera_matrix.npy', camera_matrix)
np.save('3d/dist_coeffs.npy', dist_coeffs)
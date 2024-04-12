import cv2
import numpy as np

# 准备标定板的角点坐标
# 这里假设标定板是一个9x6的棋盘格，每个方格的边长为3.5cm
pattern_size = (8, 8)
square_size = 1  # cm

# 存储角点坐标的列表
object_points = []
image_points = []

# 读取标定板图像并查找角点
for i in range(15):
    image_path = f'2d/2d-{i}.JPG'
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found:
        # 添加世界坐标系中的角点坐标
        object_points.append(np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32))
        object_points[-1][:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

        # 添加图像坐标系中的角点坐标
        image_points.append(corners)

# 进行相机校准
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# 打印相机参数
print("camera matrix:")
print(camera_matrix)
print("distortion coefficients:")
print(dist_coeffs)

# 保存相机矩阵和畸变系数为.npy文件
np.save('2d/camera_matrix.npy', camera_matrix)
np.save('2d/dist_coeffs.npy', dist_coeffs)

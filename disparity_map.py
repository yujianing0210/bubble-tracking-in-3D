import cv2

frame1 = cv2.imread('images/group1.jpg') 
frame2 = cv2.imread('images/group2.jpg')

frame1_new = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_new = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

num_disparities = 16*5
block_size = 15

sbm = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

disparity = sbm.compute(frame1_new, frame2_new)

norm_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('Disparity Map', norm_disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
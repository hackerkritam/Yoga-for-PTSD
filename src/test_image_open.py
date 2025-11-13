import cv2
img = cv2.imread('reference_mountain_pose.jpg')
if img is None:
    print("NOT FOUND")
else:
    print("FOUND") 
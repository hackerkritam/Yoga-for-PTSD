import cv2
import mediapipe as mp
import numpy as np

# Load image
img = cv2.imread('reference_mountain_pose.jpg')
if img is None:
    raise FileNotFoundError('reference_mountain_pose.jpg not found!')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)

if not results.pose_landmarks:
    print('No pose detected!')
    exit()

h, w = img.shape[:2]
landmarks = []
for lm in results.pose_landmarks.landmark:
    landmarks.append([lm.x, lm.y])

print('Extracted Landmarks:')
print('[')
for x, y in landmarks:
    print(f'    ({x:.4f}, {y:.4f}),')
print(']')

# Optional: visualize
for x, y in landmarks:
    cv2.circle(img, (int(x*w), int(y*h)), 5, (0,255,0), -1)
cv2.imshow('Reference Pose Landmarks', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
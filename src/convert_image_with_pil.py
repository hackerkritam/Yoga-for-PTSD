from PIL import Image
img = Image.open('C:/Users/KRITAM/Yoga_PTSD_System/src/reference_mountain_pose.jpg')
img.show()
img.save('C:/Users/KRITAM/Yoga_PTSD_System/src/reference_mountain_pose_cv.jpg', 'JPEG')
print("Image opened and re-saved as reference_mountain_pose_cv.jpg") 
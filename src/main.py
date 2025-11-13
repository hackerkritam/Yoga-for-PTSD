import cv2
import mediapipe as mp
import math
import time
import numpy as np
import itertools
import pyttsx3
import threading
import queue
import urllib.request
from pathlib import Path
from collections import deque


DEBUG_METRICS = False   
ACCURACY_THRESHOLD = 55.0  
SENSITIVITY = 1.08  
SMOOTHING_WINDOW = 4 


POSE_CONFIG = {
    1: {  # Mountain
        'feet_weight': 50.0,
        'shoulder_weight': 20.0,
        'arm_weight': 15.0,  # per arm
        'arm_tolerance': 40.0,
    },
    2: {  # Tree
        'leg_weight': 40.0,
        'balance_weight': 30.0,
        'standing_weight': 30.0,
    },
    3: {  # Warrior
        'leg_bend_weight': 40.0,
        'arms_weight': 30.0,
        'stance_weight': 30.0,
    },
    4: {  # Child
        'fold_weight': 50.0,
        'knee_weight': 50.0,
    },
    5: {  # Lotus
        'upright_weight': 50.0,
        'spine_weight': 30.0,
        'legs_weight': 20.0,
    }
}

speech_queue = queue.Queue()
speech_engine = None
try:
    speech_engine = pyttsx3.init()
except Exception:
    speech_engine = None

def _speech_worker():
    if speech_engine is None:
        return
    while True:
        try:
            text = speech_queue.get()
            if text is None:
                break
            speech_engine.say(text)
            speech_engine.runAndWait()
        except Exception:
            break

speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

def speak_text(text: str):
    """Non-blocking speak helper: pushes text to the speech thread (fallbacks to printing)."""
    try:
        if speech_engine is None:
            print("[SPEAK-PRINT]", text)
        else:
            speech_queue.put(text)
    except Exception:
        print("[SPEAK-ERR]", text)

def cleanup_speech():
    try:
        speech_queue.put(None)
        speech_thread.join(timeout=1.0)
    except Exception:
        pass

def speak_pose_feedback(is_correct, level, results):
    # Minimal feedback mapping; can be expanded with more detailed instructions per pose
    if is_correct:
        speak_text("Good job. Hold the pose.")
        return
    hints = {
        1: "Stand tall. Keep feet together, relax your shoulders, and raise arms slightly if needed.",
        2: "Shift weight to one leg and bring the other foot to the inner thigh. Use hands for balance.",
        3: "Bend the front knee and keep the back leg straight. Reach your arms up to increase stability.",
        4: "Kneel and fold forward, rest your forehead toward the mat and breathe deeply.",
        5: "Sit upright with a straight spine. Cross legs comfortably and rest hands on knees."
    }
    speak_text(hints.get(level, "Keep trying. Adjust your alignment and balance."))

def check_pose_accuracy(pose_level, results):
    if not results or not results.pose_landmarks:
        return False, 0.0

    accuracy = 0.0

    # compute torso normalization distance
    left_shoulder = find_point(results.pose_landmarks, 11)
    right_shoulder = find_point(results.pose_landmarks, 12)
    left_hip = find_point(results.pose_landmarks, 23)
    right_hip = find_point(results.pose_landmarks, 24)
    mid_sh = ((left_shoulder[0] + right_shoulder[0]) / 2.0, (left_shoulder[1] + right_shoulder[1]) / 2.0)
    mid_hip = ((left_hip[0] + right_hip[0]) / 2.0, (left_hip[1] + right_hip[1]) / 2.0)
    torso_dist = euclidian(mid_sh, mid_hip)
    if torso_dist < 1:
        torso_dist = max(height, width) / 4.0

    sub_scores = {}
    cfg = POSE_CONFIG.get(pose_level, {})

    if pose_level == 1:  # Mountain Pose
        left_ankle = find_point(results.pose_landmarks, 27)
        right_ankle = find_point(results.pose_landmarks, 28)
        left_wrist = find_point(results.pose_landmarks, 15)
        right_wrist = find_point(results.pose_landmarks, 16)

        shoulder_gap = abs(left_shoulder[0] - right_shoulder[0])
        ankle_gap = abs(left_ankle[0] - right_ankle[0])

        left_arm_angle = angle_calc(left_wrist, left_shoulder, find_point(results.pose_landmarks, 13))
        right_arm_angle = angle_calc(right_wrist, right_shoulder, find_point(results.pose_landmarks, 14))

        feet_w = cfg.get('feet_weight', 50.0)
        shoulder_w = cfg.get('shoulder_weight', 20.0)
        arm_w = cfg.get('arm_weight', 15.0)
        arm_tol = cfg.get('arm_tolerance', 40.0)

        feet_score = max(0.0, feet_w * (1.0 - (ankle_gap / (torso_dist * 0.6))))
        shoulder_score = max(0.0, shoulder_w * (1.0 - (shoulder_gap / (torso_dist * 0.9))))
        left_arm_score = max(0.0, arm_w * (1.0 - abs(left_arm_angle - 170) / arm_tol))
        right_arm_score = max(0.0, arm_w * (1.0 - abs(right_arm_angle - 170) / arm_tol))

        sub_scores.update({
            'feet_score': feet_score,
            'shoulder_score': shoulder_score,
            'left_arm_score': left_arm_score,
            'right_arm_score': right_arm_score,
        })

        accuracy = feet_score + shoulder_score + left_arm_score + right_arm_score
        accuracy = min(100.0, max(0.0, accuracy))

    elif pose_level == 2:  # Tree Pose
        left_hip = find_point(results.pose_landmarks, 23)
        right_hip = find_point(results.pose_landmarks, 24)
        left_ankle = find_point(results.pose_landmarks, 27)
        right_ankle = find_point(results.pose_landmarks, 28)
        left_knee = find_point(results.pose_landmarks, 25)
        right_knee = find_point(results.pose_landmarks, 26)

        hip_ankle_l = euclidian(left_hip, left_ankle)
        hip_ankle_r = euclidian(right_hip, right_ankle)

        left_leg_raised = left_ankle[1] < left_hip[1] - torso_dist * 0.35
        right_leg_raised = right_ankle[1] < right_hip[1] - torso_dist * 0.35

        if left_leg_raised:
            leg_accuracy = 40.0
            balance_diff = hip_ankle_l
        elif right_leg_raised:
            leg_accuracy = 40.0
            balance_diff = hip_ankle_r
        else:
            leg_accuracy = 0.0
            balance_diff = max(hip_ankle_l, hip_ankle_r)

        balance_accuracy = max(0.0, 30.0 * (1.0 - abs(hip_ankle_l - hip_ankle_r) / (torso_dist * 0.6)))
        standing_accuracy = max(0.0, 30.0 * (1.0 - abs(balance_diff - torso_dist * 0.75) / (torso_dist * 0.75)))

        sub_scores.update({
            'leg_accuracy': leg_accuracy,
            'balance_accuracy': balance_accuracy,
            'standing_accuracy': standing_accuracy,
        })

        accuracy = leg_accuracy + balance_accuracy + standing_accuracy
        accuracy = min(100.0, max(0.0, accuracy))

    elif pose_level == 3:  # Warrior Pose
        left_hip = find_point(results.pose_landmarks, 23)
        right_hip = find_point(results.pose_landmarks, 24)
        left_knee = find_point(results.pose_landmarks, 25)
        right_knee = find_point(results.pose_landmarks, 26)
        left_ankle = find_point(results.pose_landmarks, 27)
        right_ankle = find_point(results.pose_landmarks, 28)
        left_wrist = find_point(results.pose_landmarks, 15)
        right_wrist = find_point(results.pose_landmarks, 16)
        left_shoulder = find_point(results.pose_landmarks, 11)
        right_shoulder = find_point(results.pose_landmarks, 12)

        left_leg_angle = angle_calc(left_hip, left_knee, left_ankle)
        right_leg_angle = angle_calc(right_hip, right_knee, right_ankle)

        left_arm_raised = left_wrist[1] < left_shoulder[1] - torso_dist * 0.4
        right_arm_raised = right_wrist[1] < right_shoulder[1] - torso_dist * 0.4

        leg_bend_accuracy = max(0.0, 40.0 * (1.0 - abs(min(left_leg_angle, right_leg_angle) - 90) / 90.0))
        arms_accuracy = 30.0 if (left_arm_raised or right_arm_raised) else 0.0
        stance_accuracy = 30.0 if (abs(left_leg_angle - right_leg_angle) < 50) else 10.0

        sub_scores.update({
            'leg_bend_accuracy': leg_bend_accuracy,
            'arms_accuracy': arms_accuracy,
            'stance_accuracy': stance_accuracy,
        })

        accuracy = leg_bend_accuracy + arms_accuracy + stance_accuracy
        accuracy = min(100.0, max(0.0, accuracy))

    elif pose_level == 4:  # Child's Pose
        nose = find_point(results.pose_landmarks, 0)
        left_hip = find_point(results.pose_landmarks, 23)
        right_hip = find_point(results.pose_landmarks, 24)
        left_knee = find_point(results.pose_landmarks, 25)
        right_knee = find_point(results.pose_landmarks, 26)

        hip_center_y = (left_hip[1] + right_hip[1]) / 2

        nose_below_hips = nose[1] > hip_center_y
        knees_bent = left_knee[1] > (left_hip[1] + torso_dist * 0.2) and right_knee[1] > (right_hip[1] + torso_dist * 0.2)

        fold_accuracy = max(0.0, 50.0 * (1.0 - max(0.0, hip_center_y - nose[1]) / (torso_dist * 0.6)))
        knee_accuracy = 50.0 if knees_bent else max(0.0, 50.0 * (1.0 - abs(left_knee[1] - left_hip[1]) / (torso_dist * 0.6)))

        sub_scores.update({
            'fold_accuracy': fold_accuracy,
            'knee_accuracy': knee_accuracy,
        })

        accuracy = fold_accuracy + knee_accuracy
        accuracy = min(100.0, max(0.0, accuracy))

    elif pose_level == 5:  # Lotus Pose
        left_hip = find_point(results.pose_landmarks, 23)
        right_hip = find_point(results.pose_landmarks, 24)
        nose = find_point(results.pose_landmarks, 0)
        left_shoulder = find_point(results.pose_landmarks, 11)
        right_shoulder = find_point(results.pose_landmarks, 12)
        left_knee = find_point(results.pose_landmarks, 25)
        right_knee = find_point(results.pose_landmarks, 26)

        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

        upright_accuracy = max(0.0, 50.0 * (1.0 - abs(nose[1] - hip_center_y) / (torso_dist * 0.6)))
        spine_accuracy = max(0.0, 30.0 * (1.0 - abs(shoulder_y - nose[1]) / (torso_dist * 0.6)))
        legs_accuracy = 20.0 if (left_knee[1] > left_hip[1] and right_knee[1] > right_hip[1]) else 5.0

        sub_scores.update({
            'upright_accuracy': upright_accuracy,
            'spine_accuracy': spine_accuracy,
            'legs_accuracy': legs_accuracy,
        })

        accuracy = upright_accuracy + spine_accuracy + legs_accuracy
        accuracy = min(100.0, max(0.0, accuracy))

    is_correct = accuracy >= ACCURACY_THRESHOLD
    if DEBUG_METRICS:
        print(f"[ACCURACY] Level {pose_level}: {accuracy:.1f}% - {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")
        print("[SUBSCORES]", {k: round(v, 2) for k, v in sub_scores.items()})

    return is_correct, accuracy
    

def speak_level_complete(level):
    if level < 5:
        speak_text(f"Congratulations! Level {level} complete. Moving to level {level + 1}.")
    else:
        speak_text("Amazing! You've completed all levels. Well done!")

def find_point(landmarks, p):
    if p < len(landmarks.landmark):
        landmark = landmarks.landmark[p]
        return (int(landmark.x * width), int(landmark.y * height))
    return (0, 0)

def euclidian(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def angle_calc(p0, p1, p2):
    try:
        a = (p1[0]-p0[0])**2 + (p1[1]-p0[1])**2
        b = (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
        c = (p2[0]-p0[0])**2 + (p2[1]-p0[1])**2
        angle = math.acos((a+b-c) / math.sqrt(4*a*b)) * 180/math.pi
    except:
        return 0
    return int(angle)

def create_gradient_background(width, height, color1, color2):
    background = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        ratio = y / height
        r = int(color1[0] * (1 - ratio) + color2[0] * ratio)
        g = int(color1[1] * (1 - ratio) + color2[1] * ratio)
        b = int(color1[2] * (1 - ratio) + color2[2] * ratio)
        background[y, :] = [b, g, r]
    for i in range(0, height, 30):
        alpha = 0.05
        cv2.line(background, (0,i), (width,i), (int(255*alpha), int(255*alpha), int(255*alpha)), 1)
    return background

def draw_modern_text(img, text, position, color=(255,255,255), scale=1.0, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (position[0]+2, position[1]+2), font, scale, (0,0,0), thickness+2)
    cv2.putText(img, text, position, font, scale, color, thickness)

def draw_3d_cylinder(img, center, radius, height, color, thickness=-1):
    x, y = center
    cv2.ellipse(img, (x, y), (radius, radius//3), 0, 0, 180, color, thickness)
    cv2.ellipse(img, (x, y+height), (radius, radius//3), 0, 180, 360, color, thickness)
    cv2.rectangle(img, (x-radius, y), (x+radius, y+height), color, thickness)
    darker = tuple(max(0, c-50) for c in color)
    cv2.ellipse(img, (x+2, y), (radius, radius//3), 0, 0, 180, darker, thickness)
    cv2.ellipse(img, (x+2, y+height), (radius, radius//3), 0, 180, 360, darker, thickness)

def draw_3d_sphere(img, center, radius, color):
    x, y = center
    cv2.circle(img, (x, y), radius, color, -1)
    highlight = tuple(min(255, c+80) for c in color)
    cv2.circle(img, (x-radius//3, y-radius//3), radius//3, highlight, -1)
    shadow = tuple(max(0, c-60) for c in color)
    cv2.circle(img, (x+radius//4, y+radius//4), radius//2, shadow, -1)

def create_modern_mountain_pose():
    ref_img = create_gradient_background(640, 480, (60,80,100), (30,40,50))
    for i in range(0,480,20):
        alpha = 0.1 * (1-abs(i-240)/240)
        cv2.line(ref_img, (0,i), (640,i), (int(255*alpha), int(255*alpha), int(255*alpha)), 1)
    draw_3d_sphere(ref_img, (320,80), 25, (220,200,180))
    draw_3d_cylinder(ref_img, (320,105), 8, 20, (200,180,160))
    draw_3d_cylinder(ref_img, (320,150), 35, 80, (180,160,140))
    draw_3d_cylinder(ref_img, (290,140), 12, 60, (200,180,160))
    draw_3d_cylinder(ref_img, (350,140), 12, 60, (200,180,160))
    draw_3d_sphere(ref_img, (290,200), 15, (220,200,180))
    draw_3d_sphere(ref_img, (350,200), 15, (220,200,180))
    draw_3d_cylinder(ref_img, (300,230), 15, 90, (180,160,140))
    draw_3d_cylinder(ref_img, (340,230), 15, 90, (180,160,140))
    cv2.ellipse(ref_img, (300,320), (20,10), 0, 0, 180, (160,140,120), -1)
    cv2.ellipse(ref_img, (340,320), (20,10), 0, 0, 180, (160,140,120), -1)
    cv2.circle(ref_img, (320,200), 150, (100,150,200), 2)
    return ref_img

def create_modern_tree_pose():
    ref_img = create_gradient_background(640, 480, (80,100,60), (40,50,30))
    for i in range(0,480,20):
        alpha = 0.1 * (1-abs(i-240)/240)
        cv2.line(ref_img, (0,i), (640,i), (int(200*alpha), int(255*alpha), int(200*alpha)), 1)
    draw_3d_sphere(ref_img, (320,80), 25, (220,200,180))
    draw_3d_cylinder(ref_img, (320,105), 8, 20, (200,180,160))
    draw_3d_cylinder(ref_img, (320,150), 35, 80, (180,160,140))
    draw_3d_cylinder(ref_img, (320,140), 15, 40, (200,180,160))
    draw_3d_sphere(ref_img, (320,180), 20, (220,200,180))
    draw_3d_cylinder(ref_img, (320,230), 15, 90, (180,160,140))
    cv2.ellipse(ref_img, (320,200), (40,20), 0, 0, 180, (200,180,160), -1)
    draw_3d_sphere(ref_img, (320,200), 15, (220,200,180))
    cv2.ellipse(ref_img, (320,320), (20,10), 0, 0, 180, (160,140,120), -1)
    cv2.circle(ref_img, (320,200), 120, (150,200,100), 2)
    return ref_img

def create_modern_warrior_pose():
    ref_img = create_gradient_background(640, 480, (100,60,80), (50,30,40))
    for i in range(0,480,20):
        alpha = 0.1 * (1-abs(i-240)/240)
        cv2.line(ref_img, (0,i), (640,i), (int(255*alpha), int(200*alpha), int(220*alpha)), 1)
    draw_3d_sphere(ref_img, (320,80), 25, (220,200,180))
    draw_3d_cylinder(ref_img, (320,105), 8, 20, (200,180,160))
    draw_3d_cylinder(ref_img, (320,150), 35, 60, (180,160,140))
    draw_3d_cylinder(ref_img, (320,120), 12, 40, (200,180,160))
    draw_3d_sphere(ref_img, (320,80), 15, (220,200,180))
    draw_3d_cylinder(ref_img, (300,200), 18, 80, (180,160,140))
    cv2.ellipse(ref_img, (300,280), (25,15), 0, 0, 180, (160,140,120), -1)
    draw_3d_cylinder(ref_img, (380,200), 15, 80, (180,160,140))
    cv2.ellipse(ref_img, (380,280), (20,10), 0, 0, 180, (160,140,120), -1)
    cv2.circle(ref_img, (320,200), 130, (200,100,150), 2)
    return ref_img

def create_modern_child_pose():
    ref_img = create_gradient_background(640, 480, (60,60,100), (30,30,50))
    for i in range(0,480,20):
        alpha = 0.1 * (1-abs(i-240)/240)
        cv2.line(ref_img, (0,i), (640,i), (int(200*alpha), int(200*alpha), int(255*alpha)), 1)
    draw_3d_sphere(ref_img, (320,280), 25, (220,200,180))
    draw_3d_cylinder(ref_img, (320,305), 8, 15, (200,180,160))
    draw_3d_cylinder(ref_img, (320,320), 40, 60, (180,160,140))
    draw_3d_cylinder(ref_img, (280,320), 12, 40, (200,180,160))
    draw_3d_cylinder(ref_img, (360,320), 12, 40, (200,180,160))
    draw_3d_sphere(ref_img, (260,360), 15, (220,200,180))
    draw_3d_sphere(ref_img, (380,360), 15, (220,200,180))
    draw_3d_cylinder(ref_img, (300,380), 15, 40, (180,160,140))
    draw_3d_cylinder(ref_img, (340,380), 15, 40, (180,160,140))
    cv2.ellipse(ref_img, (300,420), (15,8), 0, 0, 180, (160,140,120), -1)
    cv2.ellipse(ref_img, (340,420), (15,8), 0, 0, 180, (160,140,120), -1)
    cv2.circle(ref_img, (320,320), 100, (100,100,200), 2)
    return ref_img

def create_modern_lotus_pose():
    ref_img = create_gradient_background(640, 480, (100,100,60), (50,50,30))
    for i in range(0,480,20):
        alpha = 0.1 * (1-abs(i-240)/240)
        cv2.line(ref_img, (0,i), (640,i), (int(255*alpha), int(255*alpha), int(200*alpha)), 1)
    draw_3d_sphere(ref_img, (320,120), 25, (220,200,180))
    draw_3d_cylinder(ref_img, (320,145), 8, 20, (200,180,160))
    draw_3d_cylinder(ref_img, (320,180), 35, 60, (180,160,140))
    draw_3d_cylinder(ref_img, (280,200), 12, 30, (200,180,160))
    draw_3d_cylinder(ref_img, (360,200), 12, 30, (200,180,160))
    draw_3d_sphere(ref_img, (280,230), 15, (220,200,180))
    draw_3d_sphere(ref_img, (360,230), 15, (220,200,180))
    cv2.ellipse(ref_img, (300,280), (30,20), 0, 0, 180, (200,180,160), -1)
    cv2.ellipse(ref_img, (340,280), (30,20), 0, 0, 180, (200,180,160), -1)
    draw_3d_sphere(ref_img, (275,280), 12, (220,200,180))
    draw_3d_sphere(ref_img, (365,280), 12, (220,200,180))
    cv2.circle(ref_img, (320,200), 110, (200,200,100), 2)
    return ref_img

def draw_animated_mountain_pose(ref_img, t):
    keyframes = get_mountain_pose_keyframes()
    n = len(keyframes)
    total_time = 2.0
    frame_idx = int(t * n) % n
    next_idx = (frame_idx + 1) % n
    local_t = (t * n) % 1.0
    angle1 = keyframes[frame_idx]['arms']
    angle2 = keyframes[next_idx]['arms']
    arms_angle = angle1 * (1 - local_t) + angle2 * local_t
    draw_3d_sphere(ref_img, (320, 80), 25, (220, 200, 180))
    draw_3d_cylinder(ref_img, (320, 105), 8, 20, (200, 180, 160))
    draw_3d_cylinder(ref_img, (320, 150), 35, 80, (180, 160, 140))
    x1, y1 = 320, 135
    length = 60
    angle_rad = math.radians(180 - arms_angle)
    x2 = int(x1 - length * math.cos(angle_rad))
    y2 = int(y1 - length * math.sin(angle_rad))
    draw_3d_cylinder(ref_img, (x1, y1), 12, int(math.hypot(x2-x1, y2-y1)), (200, 180, 160))
    draw_3d_sphere(ref_img, (x2, y2), 15, (220, 200, 180))
    angle_rad = math.radians(arms_angle)
    x2 = int(x1 + length * math.cos(angle_rad))
    y2 = int(y1 - length * math.sin(angle_rad))
    draw_3d_cylinder(ref_img, (x1, y1), 12, int(math.hypot(x2-x1, y2-y1)), (200, 180, 160))
    draw_3d_sphere(ref_img, (x2, y2), 15, (220, 200, 180))
    draw_3d_cylinder(ref_img, (300, 230), 15, 90, (180, 160, 140))
    draw_3d_cylinder(ref_img, (340, 230), 15, 90, (180, 160, 140))
    cv2.ellipse(ref_img, (300, 320), (20, 10), 0, 0, 180, (160, 140, 120), -1)
    cv2.ellipse(ref_img, (340, 320), (20, 10), 0, 0, 180, (160, 140, 120), -1)
    cv2.circle(ref_img, (320, 200), 150, (100, 150, 200), 2)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32)
]

def get_mountain_pose_skeleton_keyframes():
    return [
        [
            (0.5, 0.15), (0.5, 0.25), (0.5, 0.35), (0.5, 0.45), (0.5, 0.25), (0.5, 0.35), (0.5, 0.45),
            (0.5, 0.55), (0.5, 0.65), (0.48, 0.18), (0.52, 0.18), (0.45, 0.45), (0.55, 0.45),
            (0.43, 0.6), (0.57, 0.6), (0.41, 0.75), (0.59, 0.75), (0.41, 0.85), (0.59, 0.85),
            (0.41, 0.95), (0.59, 0.95), (0.41, 1.0), (0.59, 1.0), (0.47, 0.7), (0.53, 0.7),
            (0.47, 0.85), (0.53, 0.85), (0.47, 1.0), (0.53, 1.0), (0.47, 1.0), (0.53, 1.0),
            (0.47, 1.0), (0.53, 1.0)
        ],
        [
            (0.5, 0.15), (0.5, 0.25), (0.5, 0.35), (0.5, 0.45), (0.45, 0.15), (0.4, 0.1), (0.35, 0.05),
            (0.55, 0.15), (0.6, 0.1), (0.48, 0.18), (0.52, 0.18), (0.45, 0.45), (0.55, 0.45),
            (0.43, 0.6), (0.57, 0.6), (0.41, 0.75), (0.59, 0.75), (0.41, 0.85), (0.59, 0.85),
            (0.41, 0.95), (0.59, 0.95), (0.41, 1.0), (0.59, 1.0), (0.47, 0.7), (0.53, 0.7),
            (0.47, 0.85), (0.53, 0.85), (0.47, 1.0), (0.53, 1.0), (0.47, 1.0), (0.53, 1.0),
            (0.47, 1.0), (0.53, 1.0)
        ]
    ]

def interpolate_skeleton(kf1, kf2, t):
    return [((1-t)*x1 + t*x2, (1-t)*y1 + t*y2) for (x1, y1), (x2, y2) in zip(kf1, kf2)]

def draw_skeleton(img, landmarks, color=(0, 200, 255)):
    h, w = img.shape[:2]
    for i, j in POSE_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = (int(landmarks[i][0]*w), int(landmarks[i][1]*h))
            pt2 = (int(landmarks[j][0]*w), int(landmarks[j][1]*h))
            cv2.line(img, pt1, pt2, color, 8)
    for x, y in landmarks:
        cv2.circle(img, (int(x*w), int(y*h)), 14, (255,255,255), -1)
        cv2.circle(img, (int(x*w), int(y*h)), 10, color, -1)

def draw_realistic_skeleton(img, landmarks, color=(0, 200, 255)):
    h, w = img.shape[:2]
    torso_pts = [landmarks[11], landmarks[12], landmarks[23], landmarks[24]]
    cx = int((torso_pts[0][0] + torso_pts[1][0] + torso_pts[2][0] + torso_pts[3][0]) * w / 4)
    cy = int((torso_pts[0][1] + torso_pts[1][1] + torso_pts[2][1] + torso_pts[3][1]) * h / 4)
    cv2.ellipse(img, (cx, cy), (int(0.07*w), int(0.18*h)), 0, 0, 360, (180, 180, 200), -1)
    head_x, head_y = int(landmarks[0][0]*w), int(landmarks[0][1]*h)
    cv2.circle(img, (head_x, head_y), int(0.06*h), (220, 200, 180), -1)
    cv2.circle(img, (head_x, head_y), int(0.06*h), (0,0,0), 2)
    for i, j in POSE_CONNECTIONS:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = (int(landmarks[i][0]*w), int(landmarks[i][1]*h))
            pt2 = (int(landmarks[j][0]*w), int(landmarks[j][1]*h))
            cv2.line(img, pt1, pt2, color, 18, cv2.LINE_AA)
            cv2.line(img, (pt1[0]+4, pt1[1]+4), (pt2[0]+4, pt2[1]+4), (80,80,80), 24, cv2.LINE_AA)
    for x, y in landmarks:
        px, py = int(x*w), int(y*h)
        cv2.circle(img, (px, py), 18, (255,255,255), -1)
        cv2.circle(img, (px, py), 14, color, -1)
        cv2.circle(img, (px-4, py-4), 8, (200,200,255), -1)

mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

print("=== 🧘 YOGA PROGRESSION SYSTEM 🧘 ===")
print("Complete each pose correctly to advance to the next level!")
print("\nLevels:")
print("1. 🏔️ Mountain Pose (Tadasana) - Foundation")
print("2. 🌳 Tree Pose (Vrikshasana) - Balance")
print("3. ⚔️ Warrior Pose (Virabhadrasana) - Strength")
print("4. 👶 Child's Pose (Balasana) - Surrender")
print("5. 🪷 Lotus Pose (Padmasana) - Meditation")
print("\nPress 'q' to quit, 'r' to reset level")

speak_text("Welcome to the Yoga Progression System. Let's start with Mountain Pose. Stand straight with your feet together and arms at your sides.")

cv2.namedWindow('🧘 Yoga Progression System 🧘', cv2.WINDOW_NORMAL)
cv2.resizeWindow('🧘 Yoga Progression System 🧘', 1280, 720)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
global height, width
current_level = 1
fps_time = time.time()
pose_hold_time = 0
pose_correct = False
level_complete = False
hold_required = 3.0
pose_names = {1: "MOUNTAIN POSE (Tadasana)",2: "TREE POSE (Vrikshasana)",3: "WARRIOR POSE (Virabhadrasana)",4: "CHILD'S POSE (Balasana)",5: "LOTUS POSE (Padmasana)"}
pose_instructions = {1: "Stand straight with arms at sides, feet together",2: "Balance on one leg, place foot on opposite thigh",3: "Bend one knee deeply, extend arms overhead",4: "Kneel and fold forward, arms extended",5: "Sit cross-legged with straight spine, hands on knees"}
timing_settings = {1: 2.0,2: 3.0,3: 2.5,4: 2.0,5: 3.0}

last_incorrect_feedback_time = 0

# Friendly audio/text for poses (fallbacks in case resources aren't present)
pose_sounds = {
    1: "Starting Mountain Pose",
    2: "Starting Tree Pose",
    3: "Starting Warrior Pose",
    4: "Starting Child's Pose",
    5: "Starting Lotus Pose",
}

voice_instructions = {
    1: "Stand tall with feet together and arms at your sides.",
    2: "Balance on one leg and place the other foot on the inner thigh.",
    3: "Lunge forward with front knee bent and extend your arms.",
    4: "Kneel and fold forward, resting your forehead towards the mat.",
    5: "Sit cross-legged with a straight spine and rest your hands on your knees.",
}

# Small running history per level to smooth accuracy and reduce noise
accuracy_history = {i: deque(maxlen=SMOOTHING_WINDOW) for i in range(1, 6)}

def wrap_text(text, max_chars=40):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) + 1 <= max_chars:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())
    return lines

pose_render_prompts = {
    1: "a highly realistic 3D render of a human figure performing Mountain Pose (Tadasana), standing tall with feet together, arms relaxed by the sides, front view, studio lighting with soft shadows, detailed skin and fabric textures, neutral gradient background, confident calm expression, subtle depth of field",
    2: "a highly realistic 3D render of a human figure performing Tree Pose (Vrikshasana), balancing on one leg with the other foot placed on the inner thigh, palms together in prayer position at chest level, front view, studio lighting with soft shadows, detailed skin and fabric textures, neutral gradient background, calm facial expression, subtle depth of field",
    3: "a realistic 3D render of a human performing Warrior Pose (Virabhadrasana), front foot lunged forward with knee bent, back leg fully extended, torso upright, arms stretched overhead in a strong V shape, slightly angled three-quarter view to showcase posture depth, dramatic side lighting to emphasize muscle definition, detailed athletic wear, high-resolution textures, clean minimal background",
    4: "a realistic 3D render of a human figure performing Child's Pose (Balasana), kneeling with hips resting on heels, torso folded forward over the thighs, arms fully extended ahead on the mat, viewed from a gentle side angle, soft ambient lighting to convey relaxation, detailed fabric folds and mat texture, serene facial features, muted background colors",
    5: "a realistic 3D render of a person sitting cross-legged in Lotus Pose (Padmasana), spine tall and straight, hands resting on knees with palms up, serene facial expression, frontal camera angle at eye level, even studio lighting with subtle rim light, high-detail skin and clothing textures, neutral backdrop, meditative atmosphere with slight depth of field"
}

POSE_IMAGE_INFO = {
    1: {"filename": "mountain_poze.jpg"},
    2: {"filename": "vriksana.jpg"},
    3: {"filename": "virabhadrasna.webp"},
    4: {"filename": "balasana.jpeg"},
    5: {"filename": "padmasana.jpg"}
}

POSE_IMAGE_DIR = Path("src")

def ensure_pose_images():
    missing = []
    for info in POSE_IMAGE_INFO.values():
        path = POSE_IMAGE_DIR / info["filename"]
        if not path.exists():
            missing.append(info["filename"])
    if missing:
        print("Warning: Missing pose reference images in 'src' folder:")
        for name in missing:
            print(f" - {name}")

def load_pose_images():
    images = {}
    for level, info in POSE_IMAGE_INFO.items():
        path = POSE_IMAGE_DIR / info["filename"]
        if not path.exists():
            continue
        img = cv2.imread(str(path))
        if img is not None:
            images[level] = img
        else:
            print(f"Warning: Unable to read pose image {path}")
    return images

ensure_pose_images()
pose_images = load_pose_images()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    if results.pose_landmarks:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_point = results.pose_landmarks.landmark[connection[0]]
            end_point = results.pose_landmarks.landmark[connection[1]]
            
            start_x = int(start_point.x * width)
            start_y = int(start_point.y * height)
            end_x = int(end_point.x * width)
            end_y = int(end_point.y * height)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (50, 200, 100), 8)
        
        for landmark_id, landmark in enumerate(results.pose_landmarks.landmark):
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            if landmark_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                cv2.circle(frame, (x, y), 18, (255, 100, 200), -1)
                cv2.circle(frame, (x, y), 18, (255, 255, 255), 4)
            else:
                cv2.circle(frame, (x, y), 15, (100, 255, 200), -1)
                cv2.circle(frame, (x, y), 15, (255, 255, 255), 3)
    
    # get raw accuracy from frame
    is_correct_raw, accuracy = check_pose_accuracy(current_level, results)

    # If no landmarks, treat as zero accuracy
    if not results.pose_landmarks:
        is_correct_raw = False
        accuracy = 0.0
        print(f"[DEBUG] No pose landmarks detected")
    # append to smoothing history and compute smoothed accuracy (apply sensitivity calibration)
    adjusted_accuracy = min(100.0, float(accuracy) * SENSITIVITY)
    accuracy_history[current_level].append(adjusted_accuracy)
    smoothed_accuracy = sum(accuracy_history[current_level]) / len(accuracy_history[current_level])
    is_correct = smoothed_accuracy >= ACCURACY_THRESHOLD
    print(f"[DEBUG] Level {current_level} - raw: {accuracy:.1f}%, adjusted: {adjusted_accuracy:.1f}%, smoothed: {smoothed_accuracy:.1f}% -> {'CORRECT' if is_correct else 'INCORRECT'}")
    base_pose_img = pose_images.get(current_level)
    if base_pose_img is not None:
        ref_img = cv2.resize(base_pose_img, (width, height))
    else:
        # If image not available, show a simple gradient background with message
        ref_img = create_gradient_background(width, height, (40,60,80), (20,30,40))
        draw_modern_text(ref_img, "Pose image loading...", (width//2-150, height//2), (255,255,255), 1.5, 3)

    instruction_start_y = max(int(height * 0.55), 220)
    panel_top = max(instruction_start_y - 90, 0)
    overlay = ref_img.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height), (0,0,0), -1)
    ref_img = cv2.addWeighted(overlay, 0.35, ref_img, 0.65, 0)

    draw_modern_text(ref_img, f"LEVEL {current_level}", (50,70), (255,255,0), 1.8, 3)
    draw_modern_text(ref_img, pose_names[current_level], (50,110), (100,255,200), 1.4, 2)

    instruction = pose_instructions[current_level]
    instruction_lines = wrap_text(instruction, 35)
    for i, line in enumerate(instruction_lines):
        draw_modern_text(ref_img, line, (50, instruction_start_y + i*28), (220,220,220), 0.85, 2)

    draw_modern_text(ref_img, f"Hold for {timing_settings[current_level]}s", (50, instruction_start_y + len(instruction_lines)*28 + 20), (255,200,100), 1.1, 2)

    cv2.rectangle(ref_img, (0,0), (ref_img.shape[1]-1, ref_img.shape[0]-1), (255,255,255), 8)
    
    if is_correct:
        cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0, 255, 0), 15)
        if not pose_correct:
            pose_hold_time = time.time()
            pose_correct = True
            speak_pose_feedback(True, current_level, results)
        hold_duration = time.time() - pose_hold_time
        required_time = timing_settings[current_level]
        time_remaining = max(0, required_time - hold_duration)
        progress = min(100, (hold_duration / required_time) * 100)

        draw_modern_text(frame, f"Accuracy: {smoothed_accuracy:.0f}%", (width//2-120, height//2-120), (100,255,100), 1.4, 2)

        if hold_duration >= required_time and not level_complete:
            level_complete = True
            speak_level_complete(current_level)
            if current_level < 5:
                current_level += 1
                pose_correct = False
                level_complete = False
                print(f"🎉 LEVEL {current_level-1} COMPLETE! Moving to Level {current_level}")
                speak_text(pose_sounds[current_level])
                speak_text(voice_instructions[current_level])
                for i in range(10):
                    cv2.circle(frame, (width//2, height//2), 50 + i*10, (0,255,0), 3)
            else:
                # Celebrate and restart the progression instead of quitting the program
                print("🎉 CONGRATULATIONS! You've completed all levels! Restarting from Level 1.")
                speak_text("Congratulations! You have completed all levels. Restarting from Level 1.")
                # celebratory animation
                for i in range(20):
                    cv2.circle(frame, (width//2, height//2), 50 + i*8, (0,255,0), 3)
                draw_modern_text(frame, "🎉 ALL LEVELS COMPLETE! Restarting... 🎉", (width//2-260, height//2), (0,255,0), 1.3, 2)
                # reset to initial state so user can continue practicing
                current_level = 1
                pose_correct = False
                level_complete = False
        else:
            if time_remaining > 0:
                draw_modern_text(frame, f"Hold: {time_remaining:.1f}s", (width//2-120, height//2-40), (100,255,100), 1.5, 3)
            else:
                draw_modern_text(frame, "Perfect! Hold this pose!", (width//2-180, height//2-40), (100,255,100), 1.5, 3)
        last_incorrect_feedback_time = 0
    else:
        cv2.rectangle(frame, (0,0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 15)
        now = time.time()
        if pose_correct:
            speak_pose_feedback(False, current_level, results)
            last_incorrect_feedback_time = now
        elif last_incorrect_feedback_time == 0 or (now - last_incorrect_feedback_time) > 2.0:
            speak_pose_feedback(False, current_level, results)
            last_incorrect_feedback_time = now
        pose_correct = False
        level_complete = False
        draw_modern_text(frame, f"Accuracy: {smoothed_accuracy:.0f}%", (width//2-120, height//2-120), (100,150,255), 1.4, 2)
        draw_modern_text(frame, "Adjust your pose", (width//2-140, height//2-40), (100,200,255), 1.3, 2)
    
    fps = 1.0 / (time.time() - fps_time)
    draw_modern_text(frame, f"FPS: {fps:.1f}", (width-150, height-40), (100,200,255), 1.0, 2)
    draw_modern_text(frame, "Press 'r' to reset, 'q' to quit", (width-280, height-80), (200,200,200), 0.8, 2)
    
    draw_modern_text(frame, f"Level {current_level}", (50, 70), (255,255,100), 1.3, 2)
    draw_modern_text(frame, pose_names[current_level], (50, 105), (100,255,200), 1.1, 2)
    combined_img = np.hstack((frame, ref_img))
    cv2.rectangle(combined_img, (0,0), (combined_img.shape[1]-1, combined_img.shape[0]-1), (100,150,200), 2)
    cv2.line(combined_img, (width, 0), (width, height), (200,200,200), 4)
    cv2.rectangle(combined_img, (0, 0), (combined_img.shape[1], 45), (30,30,40), -1)
    draw_modern_text(combined_img, "🧘 YOGA PROGRESSION SYSTEM 🧘", (combined_img.shape[1]//2-240, 28), (100,255,200), 1.1, 2)
    
    cv2.imshow('🧘 Yoga Progression System 🧘', combined_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        current_level = 1
        pose_correct = False
        level_complete = False
        print("Reset to Level 1")
        speak_text("Reset to Level 1. Mountain Pose. Stand straight with your feet together and arms at your sides.")
    elif key == ord('d'):
        # toggle verbose metric logging
        DEBUG_METRICS = not DEBUG_METRICS
        print(f"DEBUG_METRICS = {DEBUG_METRICS}")
        speak_text(f"Debug metrics {'enabled' if DEBUG_METRICS else 'disabled'}")
    elif key in [ord(str(i)) for i in range(1,6)]:
        new_level = int(chr(key))
        if 1 <= new_level <= 5:
            current_level = new_level
            pose_correct = False
            level_complete = False
            print(f"Manually switched to Level {current_level}")
            speak_text(pose_sounds[current_level])
            speak_text(voice_instructions[current_level])
cap.release()
cv2.destroyAllWindows() 
cleanup_speech() 
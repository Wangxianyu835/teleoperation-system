import cv2
import mediapipe as mp
import numpy as np
import time
import urllib.request
import os
import h5py
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ====================== 模型下载 ======================
HAND_MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(HAND_MODEL_PATH):
    print("正在下载手部模型...")
    url = "https://hf-mirror.com/google/mediapipe-models/resolve/main/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, HAND_MODEL_PATH)

POSE_MODEL_PATH = 'pose_landmarker_lite.task'
if not os.path.exists(POSE_MODEL_PATH):
    print("正在下载姿态模型...")
    url = "https://hf-mirror.com/google/mediapipe-models/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)

# ====================== 检测器初始化 ======================
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.7,
    min_pose_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# ====================== 关键点筛选：只保留肩膀、手臂、手腕 ======================
SHOULDER_ARM_WRIST_IDX = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
# 骨架连线
SHOULDER_ARM_WRIST_CONNECTIONS = [
    (0, 1),    # 左肩-右肩
    (0, 2),    # 左肩 - 左肘
    (2, 4),    # 左肘 - 左腕
    (4, 6),    # 左腕 - 左手掌连接点1
    (4, 8),    # 左腕 - 左手掌连接点2
    (4, 10),   # 左腕 - 左手掌连接点3
    (1, 3),    # 右肩 - 右肘
    (3, 5),    # 右肘 - 右腕
    (5, 7),    # 右腕 - 右手掌连接点1
    (5, 9),    # 右腕 - 右手掌连接点2
    (5, 11),   # 右腕 - 右手掌连接点3
    (0, 12),   # 左肩 - 左髋
    (1, 13),   # 右肩 - 右髋
    (12, 13),  # 左髋 - 右髋
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ====================== 摄像头 ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ====================== 存储 ======================
frame_id_list = []
timestamp_list = []
pose_kps_list = []
pose_visibility_list = []
left_hand_kps_list = []
right_hand_kps_list = []

pose_valid_list = []
left_hand_valid_list = []
right_hand_valid_list = []

start_time = time.time()
VIS_THRESHOLD = 0.5

print("="*60)
print("【防假肢体优化版】单只手时自动屏蔽另一侧手臂")
print("按 q 停止并保存 H5")
print("="*60)

with HandLandmarker.create_from_options(hand_options) as hand_landmarker, \
     PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("摄像头读取失败！")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        pose_kp = np.zeros((14, 3), dtype=np.float32)
        pose_vis = np.zeros((14,), dtype=np.float32)
        pose_valid = False
        left_kp = np.zeros((21, 3), dtype=np.float32)
        right_kp = np.zeros((21, 3), dtype=np.float32)
        left_hand_present = False
        right_hand_present = False

        h, w, _ = frame.shape

        # ========= 1. 解析手部，标记左右手是否存在 =========
        if hand_result.hand_landmarks:
            for i, hand_landmarks in enumerate(hand_result.hand_landmarks):
                hand_type = hand_result.handedness[i][0].category_name
                kp_arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                if hand_type == "Left":
                    left_kp = kp_arr
                    left_hand_present = True
                else:
                    right_kp = kp_arr
                    right_hand_present = True

                for (s, e) in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[s].x * w)
                    y1 = int(hand_landmarks[s].y * h)
                    x2 = int(hand_landmarks[e].x * w)
                    y2 = int(hand_landmarks[e].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # ========= 2. 解析Pose关键点 + 【单侧肢体屏蔽核心逻辑】 =========
        if pose_result.pose_landmarks:
            full_pose = pose_result.pose_landmarks[0]
            raw_pose = []
            raw_vis = []
            for idx in SHOULDER_ARM_WRIST_IDX:
                lm = full_pose[idx]
                raw_pose.append([lm.x, lm.y, lm.z])
                raw_vis.append(lm.visibility)
            raw_pose = np.array(raw_pose, dtype=np.float32)
            raw_vis = np.array(raw_vis, dtype=np.float32)

            # 低置信点置零
            mask = raw_vis < VIS_THRESHOLD
            raw_pose[mask, :] = 0.0

            # ===== 关键：根据检测到的手，关闭不存在那一侧的手臂 =====
            if left_hand_present and (not right_hand_present):
                # 只有左手：清空右侧肢体点 [1,3,5,7,9,11]
                raw_pose[[1,3,5,7,9,11], :] = 0.0
                raw_vis[[1,3,5,7,9,11]] = 0.0
            elif right_hand_present and (not left_hand_present):
                # 只有右手：清空左侧肢体点 [0,2,4,6,8,10]
                raw_pose[[0,2,4,6,8,10], :] = 0.0
                raw_vis[[0,2,4,6,8,10]] = 0.0
            elif (not left_hand_present) and (not right_hand_present):
                # 两只手都没检测到：整个pose全部清零，完全不显示骨架
                raw_pose[:,:] = 0.0
                raw_vis[:] = 0.0

            pose_kp = raw_pose
            pose_vis = raw_vis
            # 只要还有有效点就标记pose有效
            pose_valid = np.sum(np.abs(pose_kp))>1e-6

            # 绘制筛选后的骨架
            for (s, e) in SHOULDER_ARM_WRIST_CONNECTIONS:
                if pose_vis[s] < VIS_THRESHOLD or pose_vis[e] < VIS_THRESHOLD:
                    continue
                x1 = int(pose_kp[s, 0] * w)
                y1 = int(pose_kp[s, 1] * h)
                x2 = int(pose_kp[e, 0] * w)
                y2 = int(pose_kp[e, 1] * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255,255,0), 2)
            for idx in range(pose_kp.shape[0]):
                if pose_vis[idx] >= VIS_THRESHOLD:
                    cx = int(pose_kp[idx,0] * w)
                    cy = int(pose_kp[idx,1] * h)
                    cv2.circle(frame, (cx, cy), 4, (0,255,255), -1)

        # ========= 3. 保存标记 =========
        frame_id_list.append(frame_id)
        timestamp_list.append(round(time.time() - start_time, 3))
        pose_kps_list.append(pose_kp)
        pose_visibility_list.append(pose_vis)
        left_hand_kps_list.append(left_kp)
        right_hand_kps_list.append(right_kp)

        pose_valid_list.append(pose_valid)
        left_hand_valid_list.append(left_hand_present)
        right_hand_valid_list.append(right_hand_present)

        cv2.putText(frame, f"Frame:{frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Shoulder-Arm-Wrist + Hands Capture | SingleArm Suppress", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_id += 1

cap.release()
cv2.destroyAllWindows()

# ====================== 保存 H5 ======================
save_dir = "./body_hand_arm_dataset"
os.makedirs(save_dir, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"arm_hand_data_{time_str}.h5")

frame_ids = np.array(frame_id_list)
timestamps = np.array(timestamp_list)
pose_data = np.array(pose_kps_list)
pose_vis_data = np.array(pose_visibility_list)
left_hand_data = np.array(left_hand_kps_list)
right_hand_data = np.array(right_hand_kps_list)

pose_valid_arr = np.array(pose_valid_list, dtype=bool)
left_valid_arr = np.array(left_hand_valid_list, dtype=bool)
right_valid_arr = np.array(right_hand_valid_list, dtype=bool)

with h5py.File(save_path, 'w') as f:
    f.create_dataset("frame_ids", data=frame_ids)
    f.create_dataset("timestamps", data=timestamps)
    f.create_dataset("pose_keypoints", data=pose_data)
    f.create_dataset("pose_visibility", data=pose_vis_data)
    f.create_dataset("left_hand_keypoints", data=left_hand_data)
    f.create_dataset("right_hand_keypoints", data=right_hand_data)
    f.create_dataset("pose_valid", data=pose_valid_arr)
    f.create_dataset("left_hand_valid", data=left_valid_arr)
    f.create_dataset("right_hand_valid", data=right_valid_arr)

    f.attrs["data_source"] = "MediaPipe Shoulder-Arm-Wrist + Hands | Single arm suppress"
    f.attrs["collect_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    f.attrs["frame_total"] = len(frame_ids)
    f.attrs["pose_keypoint_count"] = 14
    f.attrs["vis_threshold"] = VIS_THRESHOLD

print("\n" + "="*60)
print("✅ 采集完成，已开启单侧肢体抑制")
print(f"总帧数：{len(frame_ids)}")
print(f"姿态有效帧：{np.sum(pose_valid_arr)}")
print(f"左手有效帧：{np.sum(left_valid_arr)}")
print(f"右手有效帧：{np.sum(right_valid_arr)}")
print(f"保存路径：{save_path}")
print("="*60)


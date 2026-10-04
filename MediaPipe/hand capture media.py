'''
MediaPipe 视觉纯手部数据采集
功能：摄像头实时采集双手21关键点XYZ、可视化骨架、保存标准H5数据集
适配：与VisionPro数据集格式统一，可直接用于后续模型训练
优化：解决国内模型下载失败、帧率异常、数据冗余问题
'''
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import h5py
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ====================== 核心配置（可直接改） ======================
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAM_FPS = 30
MIN_DET_CONF = 0.5
MIN_TRK_CONF = 0.5

# ====================== 解决国内无法自动下载模型问题 ======================
MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):


    import urllib.request
    url = "https://hf-mirror.com/google/mediapipe-models/resolve/main/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("模型下载完成！")

# ====================== 初始化MediaPipe手部模型 ======================
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=MIN_DET_CONF,
    min_hand_presence_confidence=MIN_DET_CONF,
    min_tracking_confidence=MIN_TRK_CONF
)

# 手部骨架连线
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ====================== 初始化摄像头 ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

# ====================== 数据存储容器（规范格式） ======================
frame_id_list = []
timestamp_list = []
left_hand_kps_list = []   # 左手 (21,3) 无手则全0
right_hand_kps_list = []  # 右手 (21,3) 无手则全0

start_time = time.time()

print("="*60)
print("MediaPipe 视觉手部3D关键点采集程序启动成功！")
print("操作方式：摄像头对准手部，按【q键】停止并自动保存H5数据集")
print("="*60)

# ====================== 主采集循环 ======================
with HandLandmarker.create_from_options(options) as landmarker:
    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("摄像头读取失败，请检查设备占用情况！")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 初始化单帧数据：无手填充0，保证维度统一
        left_kp = np.zeros((21, 3), dtype=np.float32)
        right_kp = np.zeros((21, 3), dtype=np.float32)

        # 解析双手数据
        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                hand_type = result.handedness[i][0].category_name
                kp_array = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
                if hand_type == "Left":
                    left_kp = kp_array
                else:
                    right_kp = kp_array

                # 可视化关键点编号
                h, w, _ = frame.shape
                for idx, lm in enumerate(hand_landmarks):
                    cx = int(lm.x * w)
                    cy = int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (cx+4, cy-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1)
                # 可视化骨架
                for (s, e) in HAND_CONNECTIONS:
                    x1 = int(hand_landmarks[s].x * w)
                    y1 = int(hand_landmarks[s].y * h)
                    x2 = int(hand_landmarks[e].x * w)
                    y2 = int(hand_landmarks[e].y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 存入数据集
        frame_id_list.append(frame_id)
        timestamp_list.append(round(time.time() - start_time, 3))
        left_hand_kps_list.append(left_kp)
        right_hand_kps_list.append(right_kp)

        # 稳定FPS显示
        real_fps = round(1.0 / (time.time() - start_time + 1e-6), 1)
        cv2.putText(frame, f"FPS: {real_fps} | Frame: {frame_id}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("MediaPipe Hand Capture (Press q to exit)", frame)

        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

# ====================== 释放资源 ======================
cap.release()
cv2.destroyAllWindows()

# ====================== 统一保存为H5数据集（和VR格式对齐） ======================
save_dir = "./visual_hand_dataset"
os.makedirs(save_dir, exist_ok=True)
time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"visual_hand_data_{time_str}.h5")

# 转为数组
frame_ids = np.array(frame_id_list)
timestamps = np.array(timestamp_list)
left_data = np.array(left_hand_kps_list)
right_data = np.array(right_hand_kps_list)

# 写入H5
with h5py.File(save_path, 'w') as f:
    f.create_dataset('frame_ids', data=frame_ids)
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('left_hand_keypoints', data=left_data)   # (N,21,3)
    f.create_dataset('right_hand_keypoints', data=right_data) # (N,21,3)
    # 数据集属性
    f.attrs['data_source'] = 'MediaPipe_Vision_Camera'
    f.attrs['frame_count'] = len(frame_ids)
    f.attrs['collect_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 统计有效帧
valid_cnt = np.sum([np.sum(lf)>0 or np.sum(rf)>0 for lf,rf in zip(left_data,right_data)])

print("\n" + "="*60)
print(f"✅ 视觉手部数据采集完成！")
print(f"总帧数：{len(frame_ids)}")
print(f"有效手部帧数：{valid_cnt}")
print(f"数据保存路径：{save_path}")
print("="*60)

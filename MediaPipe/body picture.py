import h5py
import numpy as np
import cv2
import os
from glob import glob

# ===================== 关键点索引和连线，和采集代码完全保持一致 =====================
# 采集代码保留的14个姿态点：[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
POSE_CONNECTIONS = [
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

# ===================== 自动读取最新h5文件 =====================
def get_latest_h5():
    file_list = sorted(glob("./body_hand_arm_dataset/*.h5"), key=os.path.getmtime)
    if len(file_list) == 0:
        print("❌ 未找到肩臂手部H5数据集！请确认采集代码已经运行并保存")
        exit()
    return file_list[-1]

h5_path = get_latest_h5()
print(f"✅ 正在加载数据集：{h5_path}")

# ===================== 读取H5 =====================
with h5py.File(h5_path, 'r') as f:
    frame_ids = f['frame_ids'][:]
    timestamps = f['timestamps'][:]
    pose_kp = f['pose_keypoints'][:]         # (N,14,3)
    left_hand_kp = f['left_hand_keypoints'][:] # (N,21,3)
    right_hand_kp = f['right_hand_keypoints'][:]# (N,21,3)
    pose_valid = f['pose_valid'][:]
    left_valid = f['left_hand_valid'][:]
    right_valid = f['right_hand_valid'][:]
    attrs = dict(f.attrs)

# ===================== 基础校验 =====================
print("\n" + "="*60)
print("【数据集信息】")
print(f"总帧数：{len(frame_ids)}")
print(f"姿态关键点维度：{pose_kp.shape}")
print(f"左手关键点维度：{left_hand_kp.shape}")
print(f"右手关键点维度：{right_hand_kp.shape}")
print(f"采集时间：{attrs['collect_time']}")
print(f"姿态有效帧：{np.sum(pose_valid)}")
print(f"左手有效帧：{np.sum(left_valid)}")
print(f"右手有效帧：{np.sum(right_valid)}")

assert pose_kp.shape[1:] == (14,3), "❌ 姿态点维度错误，应为(14,3)"
assert left_hand_kp.shape[1:] == (21,3), "❌ 左手关键点维度错误"
assert right_hand_kp.shape[1:] == (21,3), "❌ 右手关键点维度错误"
print("✅ 维度校验通过")

# 判断关键点是否有效
def is_valid(keypoint_frame):
    return np.sum(np.abs(keypoint_frame)) > 1e-6

# ===================== 回放 =====================
print("\n🎬 开始回放骨架，按 q 退出回放窗口")
img_h, img_w = 480, 640

for i in range(len(frame_ids)):
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    pose_f = pose_kp[i]
    lh_f = left_hand_kp[i]
    rh_f = right_hand_kp[i]

    # 绘制肩臂骨架（黄色）
    if is_valid(pose_f):
        for (s,e) in POSE_CONNECTIONS:
            x1 = int(pose_f[s,0] * img_w)
            y1 = int(pose_f[s,1] * img_h)
            x2 = int(pose_f[e,0] * img_w)
            y2 = int(pose_f[e,1] * img_h)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,255,0), 2)
        for idx in range(pose_f.shape[0]):
            x = int(pose_f[idx,0] * img_w)
            y = int(pose_f[idx,1] * img_h)
            cv2.circle(canvas, (x,y), 4, (0,255,255), -1)

    # 左手 蓝色
    if is_valid(lh_f):
        for (s,e) in HAND_CONNECTIONS:
            x1 = int(lh_f[s,0] * img_w)
            y1 = int(lh_f[s,1] * img_h)
            x2 = int(lh_f[e,0] * img_w)
            y2 = int(lh_f[e,1] * img_h)
            cv2.line(canvas, (x1,y1), (x2,y2), (255,0,0), 2)
        for idx in range(21):
            x,y = int(lh_f[idx,0]*img_w), int(lh_f[idx,1]*img_h)
            cv2.circle(canvas, (x,y), 4, (255,0,0), -1)

    # 右手 绿色
    if is_valid(rh_f):
        for (s,e) in HAND_CONNECTIONS:
            x1 = int(rh_f[s,0] * img_w)
            y1 = int(rh_f[s,1] * img_h)
            x2 = int(rh_f[e,0] * img_w)
            y2 = int(rh_f[e,1] * img_h)
            cv2.line(canvas, (x1,y1), (x2,y2), (0,255,0), 2)
        for idx in range(21):
            x,y = int(rh_f[idx,0]*img_w), int(rh_f[idx,1]*img_h)
            cv2.circle(canvas, (x,y), 4, (0,255,0), -1)

    cv2.putText(canvas, f"Frame:{i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.imshow("H5 Shoulder-Arm-Wrist + Hand Validation", canvas)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\n✅ 校验结束，数据集结构正常，可以交付训练")

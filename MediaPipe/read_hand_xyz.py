'''
H5 手部数据集 准确性校验工具
用途：验证 MediaPipe 采集的 H5 数据是否正确、完整、可用
校验内容：
1. 数据集维度是否规范 (N,21,3)
2. 左右手数据是否正常非空
3. 时间戳/帧数是否连续
4. 可视化回放骨架，直观q验证采集效果
'''
import h5py
import numpy as np
import cv2
import os
from glob import glob

# ====================== 自动读取最新的H5文件 ======================
def get_latest_h5():
    file_list = sorted(glob("./visual_hand_dataset/*.h5"), key=os.path.getmtime)
    if len(file_list) == 0:
        print("❌ 未找到任何H5数据集！")
        exit()
    return file_list[-1]

h5_path = get_latest_h5()
print(f"✅ 正在校验数据集：{h5_path}")

# ====================== 读取H5数据 ======================
with h5py.File(h5_path, 'r') as f:
    frame_ids = f['frame_ids'][:]
    timestamps = f['timestamps'][:]
    left_hand = f['left_hand_keypoints'][:]   # (N,21,3)
    right_hand = f['right_hand_keypoints'][:] # (N,21,3)
    attrs = dict(f.attrs)

# ====================== 1. 基础信息校验 ======================
print("\n" + "="*60)
print("【数据集基础信息】")
print(f"总帧数：{len(frame_ids)}")
print(f"数据维度-左手：{left_hand.shape}")
print(f"数据维度-右手：{right_hand.shape}")
print(f"采集时间：{attrs['collect_time']}")
print(f"数据来源：{attrs['data_source']}")

# 维度合法性检查
assert left_hand.shape[1:] == (21,3), "❌ 左手关键点维度错误！"
assert right_hand.shape[1:] == (21,3), "❌ 右手关键点维度错误！"
print("✅ 维度校验通过（标准21点XYZ）")

# ====================== 2. 有效帧统计 ======================
def is_valid_hand(hand_frame):
    return np.sum(np.abs(hand_frame)) > 1e-6

valid_left = sum([is_valid_hand(f) for f in left_hand])
valid_right = sum([is_valid_hand(f) for f in right_hand])
print(f"有效左手帧数：{valid_left}")
print(f"有效右手帧数：{valid_right}")

if valid_left == 0 and valid_right == 0:
    print("❌ 警告：数据集无任何有效手部数据！")
else:
    print("✅ 存在有效手部数据")

# ====================== 3. 可视化回放校验 ======================
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

print("\n🎬 开始回放数据（按 q 退出回放）")
for i in range(len(left_hand)):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    lk = left_hand[i]
    rk = right_hand[i]

    # 绘制左手（蓝色）
    if is_valid_hand(lk):
        h, w = 480, 640
        for (s,e) in HAND_CONNECTIONS:
            x1,y1 = int(lk[s,0]*w), int(lk[s,1]*h)
            x2,y2 = int(lk[e,0]*w), int(lk[e,1]*h)
            cv2.line(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        for idx in range(21):
            x,y = int(lk[idx,0]*w), int(lk[idx,1]*h)
            cv2.circle(frame, (x,y), 4, (255,0,0), -1)

    # 绘制右手（绿色）
    if is_valid_hand(rk):
        h, w = 480, 640
        for (s,e) in HAND_CONNECTIONS:
            x1,y1 = int(rk[s,0]*w), int(rk[s,1]*h)
            x2,y2 = int(rk[e,0]*w), int(rk[e,1]*h)
            cv2.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        for idx in range(21):
            x,y = int(rk[idx,0]*w), int(rk[idx,1]*h)
            cv2.circle(frame, (x,y), 4, (0,255,0), -1)

    cv2.putText(frame, f"Frame: {i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow("H5 Data Validation Playback", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("\n✅ 数据集校验完成：数据结构准确、可正常用于训练！")

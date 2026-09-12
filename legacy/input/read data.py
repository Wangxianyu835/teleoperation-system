import numpy as np

# ====================== 1. 加载原始数据 ======================
# 加载数据（确保路径正确）
data = np.load("hand_3d_keypoints.npy", allow_pickle=True)

# 查看基础信息
print("=" * 50)
print("手部数据读取结果：")
print(f"✅ 有效采集帧数：{len(data)}")
print(f"✅ 单帧数据包含字段：{list(data[0].keys())}")  # frame_id/timestamp/left_hand/right_hand

# 查看具体的关键点数据（找第一帧有手部的画面）
has_hand_frame = None
for i, frame in enumerate(data):
    if frame["left_hand"] is not None or frame["right_hand"] is not None:
        has_hand_frame = frame
        frame_idx = i
        break

if has_hand_frame is not None:
    print(f"\n📌 第 {frame_idx} 帧（首个有手部的帧）：")
    # 查看左手关键点（如果存在）
    if has_hand_frame["left_hand"] is not None:
        print(f"   左手关键点数量：{len(has_hand_frame['left_hand'])} 个")
        print(f"   左手关键点形状：{has_hand_frame['left_hand'].shape}")  # 应该是 (21, 3)
        print(f"   第一个左手关键点坐标：{has_hand_frame['left_hand'][0]}")  # 第一个关键点（手腕）的3D坐标
    # 查看右手关键点（如果有）
    if has_hand_frame["right_hand"] is not None:
        print(f"   右手关键点数量：{len(has_hand_frame['right_hand'])} 个")
        print(f"   右手关键点形状：{has_hand_frame['right_hand'].shape}")
        print(f"   第一个右手关键点坐标：{has_hand_frame['right_hand'][0]}")
else:
    print("\n⚠️  未检测到有效手部数据，请检查采集时是否有手出现在摄像头前")

# ====================== 2. 数据统计 ======================
left_hand_count = sum(1 for frame in data if frame["left_hand"] is not None)
right_hand_count = sum(1 for frame in data if frame["right_hand"] is not None)
print(f"\n📊 数据统计：")
print(f"   左手出现帧数：{left_hand_count}")
print(f"   右手出现帧数：{right_hand_count}")
print("=" * 50)


# ====================== 3. 增强版数据平滑去噪（核心优化） ======================
# 定义中位数滤波+滑动平均平滑函数（去噪效果更明显）
def smooth_keypoints(keypoints_list, window_size=7):
    if len(keypoints_list) < window_size:
        return keypoints_list  # 数据太少时不处理
    smoothed = []
    for i in range(len(keypoints_list)):
        # 取当前帧前后 window_size//2 帧的中位数（抗极端噪声）
        start = max(0, i - window_size // 2)
        end = min(len(keypoints_list), i + window_size // 2 + 1)
        window = np.array(keypoints_list[start:end])
        smoothed_kp = np.median(window, axis=0)  # 中位数滤波（比平均值更稳）
        smoothed.append(smoothed_kp)
    return np.array(smoothed)


# 分别对左手/右手关键点做平滑
smoothed_data = []
for frame in data:
    new_frame = frame.copy()  # 保留原数据结构

    # 平滑左手关键点
    if frame["left_hand"] is not None:
        # 先收集所有左手帧的关键点，再整体平滑
        left_keypoints = [f["left_hand"] for f in data if f["left_hand"] is not None]
        if len(left_keypoints) > 0:
            smoothed_left = smooth_keypoints(left_keypoints)
            # 找到当前帧在左手序列中的位置，赋值回去
            left_indices = [i for i, f in enumerate(data) if f["left_hand"] is not None]
            curr_idx = [i for i, f in enumerate(data) if f is frame][0]
            if curr_idx in left_indices:
                pos = left_indices.index(curr_idx)
                new_frame["left_hand"] = smoothed_left[pos]

    # 平滑右手关键点（逻辑同上）
    if frame["right_hand"] is not None:
        right_keypoints = [f["right_hand"] for f in data if f["right_hand"] is not None]
        if len(right_keypoints) > 0:
            smoothed_right = smooth_keypoints(right_keypoints)
            right_indices = [i for i, f in enumerate(data) if f["right_hand"] is not None]
            curr_idx = [i for i, f in enumerate(data) if f is frame][0]
            if curr_idx in right_indices:
                pos = right_indices.index(curr_idx)
                new_frame["right_hand"] = smoothed_right[pos]

    smoothed_data.append(new_frame)

# ====================== 4. 验证平滑效果（直观对比） ======================
if has_hand_frame is not None:
    # 取前10帧食指尖（第8个关键点）的x坐标对比
    original_tip = []
    smoothed_tip = []
    for i, (orig, smooth) in enumerate(zip(data, smoothed_data)):
        if i >= 10:
            break
        if orig["left_hand"] is not None and smooth["left_hand"] is not None:
            original_tip.append(orig["left_hand"][8, 0])
            smoothed_tip.append(smooth["left_hand"][8, 0])

    print("\n【平滑前后对比（前10帧食指尖x坐标）】")
    print(f"原始数据（波动大）：{[round(x, 4) for x in original_tip]}")
    print(f"平滑数据（更稳定）：{[round(x, 4) for x in smoothed_tip]}")

# ====================== 5. 保存平滑后的数据（覆盖旧文件，强制更新） ======================
np.save("smoothed_hand_data.npy", smoothed_data)
print("\n" + "=" * 50)
print("✅ 数据平滑去噪完成！")
print("📁 新数据文件：smoothed_hand_data.npy（已覆盖旧文件）")
print("💡 提示：接下来运行图片生成脚本会自动使用这个最新的去噪后的数据！")
print("=" * 50)

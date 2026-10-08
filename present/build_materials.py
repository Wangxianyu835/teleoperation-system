"""Generate presentation media from existing data; no model/input mutation.

Run from the repository root with the validated teleoperation interpreter.
Input recordings/checkpoint are external local artifacts, not bundled here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
import torch

ROOT = Path(__file__).resolve().parents[1]
from teleoperation.simulation.environment import SimulationEnv
from teleoperation.simulation.robot_loader import RobotLoader, read_joint_ranges
from teleoperation.retargeting.hand.kinematics import create_hand_kinematics
from teleoperation.retargeting.hand.config import L21, JOINT_EDGES, JOINT_NAMES
from teleoperation.retargeting.hand.coordinates import build_l21_reference_basis
from teleoperation.apps.replay.replay_hand_native import style_hands
from teleoperation.contracts.hand import L21HandAngles
from teleoperation.robots.native_hand import NativeHandAdapter, build_mapping

OUT = Path(__file__).resolve().parent
IMAGES, VIDEOS = OUT / 'images', OUT / 'videos'
W, H = 1600, 1000
FPS = 15
BG, INK, MUTED = '#F2F5FA', '#142B48', '#586A81'
BLUE, TEAL, ORANGE = '#2878CA', '#138B87', '#D38425'
ROBOT_NAMES = {'h1_2': 'H1-2', 'gr1_t2': 'GR1-T2', 'g1': 'G1'}
SEGMENTS = {'left': (2640, 2940), 'right': (3465, 3765)}
FONT_FILE = None
FONT_CACHE = {}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def font(size=24, bold=False):
    key = (size, bold)
    if key not in FONT_CACHE:
        path = Path(FONT_FILE)
        if bold and path.name == 'msyh.ttc':
            path = path.with_name('msyhbd.ttc')
        FONT_CACHE[key] = ImageFont.truetype(str(path), size)
    return FONT_CACHE[key]


def text(im, xy, value, size=24, color=INK, bold=False):
    ImageDraw.Draw(im).text(xy, str(value), font=font(size, bold), fill=color)


def wrapped(im, xy, value, max_width, size=24, color=MUTED, gap=10):
    draw = ImageDraw.Draw(im)
    x, y = xy
    for paragraph in str(value).split('\n'):
        line = ''
        for char in paragraph:
            if draw.textlength(line + char, font=font(size)) > max_width:
                text(im, (x, y), line, size, color)
                y += size + gap
                line = char
            else:
                line += char
        text(im, (x, y), line, size, color)
        y += size + gap
    return y


def card(im, box, fill='white', outline=None):
    ImageDraw.Draw(im).rounded_rectangle(box, radius=18, fill=fill, outline=outline, width=2)


def page(title, subtitle, size=(W, H)):
    im = Image.new('RGB', size, BG)
    text(im, (50, 31), '双臂灵巧手遥操作 · 阶段成果', 18, BLUE, True)
    text(im, (50, 66), title, 39 if size[0] >= 1500 else 32, INK, True)
    text(im, (50, 126), subtitle, 21, MUTED)
    ImageDraw.Draw(im).line((50, size[1]-59, size[0]-50, size[1]-59), fill='#D5DFEB', width=2)
    text(im, (50, size[1]-43), '2026-10-07  |  PyBullet / canonical hand  |  当前项目阶段材料', 17, MUTED)
    return im


def save_image(im, name):
    im.save(IMAGES / name)
    print('IMAGE', name, flush=True)


def arrow(im, a, b, dashed=False, color=BLUE):
    d = ImageDraw.Draw(im)
    delta = np.asarray(b, float)-a
    length = float(np.linalg.norm(delta))
    unit = delta/max(length, 1)
    if dashed:
        for offset in np.arange(0, length, 17):
            q, r = np.asarray(a)+unit*offset, np.asarray(a)+unit*min(offset+9, length)
            d.line((*q, *r), fill=color, width=4)
    else:
        d.line((*a, *b), fill=color, width=4)
    tip = np.asarray(b)
    perp = np.array([-unit[1], unit[0]])
    d.polygon([tuple(tip), tuple(tip-unit*15+perp*7), tuple(tip-unit*15-perp*7)], fill=color)


def node(im, box, label, detail='', pending=False):
    card(im, box, '#FFF5E8' if pending else 'white', '#DEB877' if pending else '#CDDEEF')
    text(im, (box[0]+18, box[1]+20), label, 28, ORANGE if pending else BLUE, True)
    wrapped(im, (box[0]+18, box[1]+66), detail, box[2]-box[0]-36, 21)


class Scene:
    def __init__(self, robot):
        self.cid = p.connect(p.DIRECT)
        self.robot = robot
        self.rid = RobotLoader(self.cid).load_robot(robot)['robot']
        ranges, self.names = read_joint_ranges(self.rid, self.cid)
        self.adapters = {
            side: NativeHandAdapter(robot, side, ranges, limit_mode='clamp')
            for side in SEGMENTS
        }
        self.mapping = {side: adapter.mapping for side, adapter in self.adapters.items()}
        style_hands(p, self.rid, self.cid, {s: (m, None, None, self.names) for s, m in self.mapping.items()})
        self.cameras = {}
        self.neutral = {i: p.getJointState(self.rid, i, physicsClientId=self.cid)[0]
                        for i in range(p.getNumJoints(self.rid, physicsClientId=self.cid))
                        if p.getJointInfo(self.rid, i, physicsClientId=self.cid)[2] != p.JOINT_FIXED}

    def pose(self, side, angles):
        dofs = L21HandAngles(angles).native_mapping_dofs()
        targets, clipped = self.adapters[side].map(dofs)
        for name, value in targets.items():
            p.resetJointState(self.rid, self.names[name], value, physicsClientId=self.cid)
        return clipped

    def reset(self):
        for index, value in self.neutral.items():
            p.resetJointState(self.rid, index, value, physicsClientId=self.cid)

    def prepare_hand(self, side, angles):
        boxes = []
        for row in angles[np.linspace(0, len(angles)-1, 12, dtype=int)]:
            self.pose(side, row)
            boxes.extend(p.getAABB(self.rid, self.names[n], physicsClientId=self.cid)
                         for _, n, *_ in self.mapping[side])
        lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
        center = (lo+hi)/2
        distance = max(.16, float(np.linalg.norm(hi-lo))*1.48)
        yaws = {'h1_2': {'left':45,'right':135}, 'gr1_t2': {'left':315,'right':225},
                'g1': {'left':45,'right':135}}
        self.cameras[side] = (center, distance, yaws[self.robot][side], -12)
        self.reset()

    def render(self, width, height, side=None):
        if side:
            center, distance, yaw, pitch = self.cameras[side]
        else:
            boxes = [p.getAABB(self.rid, i, physicsClientId=self.cid)
                     for i in range(-1, p.getNumJoints(self.rid, physicsClientId=self.cid))]
            lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
            center = (lo+hi)/2
            distance = float(np.max(hi-lo))*1.65
            yaw, pitch = 135, -8
        return render_camera(self.cid, center, distance, yaw, pitch, width, height)

    def close(self):
        p.disconnect(self.cid)


def render_camera(cid, center, distance, yaw, pitch, width, height):
    view = p.computeViewMatrixFromYawPitchRoll(center, distance, yaw, pitch, 0, 2)
    proj = p.computeProjectionMatrixFOV(42, width/height, .01, 100)
    rgba = p.getCameraImage(width, height, viewMatrix=view, projectionMatrix=proj,
                           renderer=p.ER_TINY_RENDERER, shadow=0, lightDirection=[-1, -1, 2],
                           lightAmbientCoeff=.75, lightDiffuseCoeff=.65, physicsClientId=cid)[2]
    return Image.fromarray(np.asarray(rgba, np.uint8).reshape(height, width, 4)[:, :, :3])


def skeleton(xyz, edges, bounds, width=360, height=360):
    im = Image.new('RGB', (width, height), 'white')
    d = ImageDraw.Draw(im)
    # Both sets use the same palm basis; each panel retains its own display scale.
    low, high = bounds
    scale = min((width-70)/max(high[0]-low[0], 1e-5), (height-65)/max(high[1]-low[1], 1e-5))
    center = (low+high)/2
    xy = np.column_stack([(xyz[:, 0]-center[0])*scale+width/2,
                          -(xyz[:, 1]-center[1])*scale+height/2])
    for grid in range(35, width, 45):
        d.line((grid, 20, grid, height-20), fill='#EFF3F8')
    for grid in range(25, height, 45):
        d.line((20, grid, width-20, grid), fill='#EFF3F8')
    for a, b in edges:
        d.line((*xy[a], *xy[b]), fill=BLUE, width=4)
    for x, y in xy:
        d.ellipse((x-4, y-4, x+4, y+4), fill=ORANGE)
    return im


class Data:
    def __init__(self, angle_path, source_path=None):
        self.angle_path = Path(angle_path).resolve()
        with h5py.File(self.angle_path, 'r') as f:
            self.attrs = {k: (v.tolist() if isinstance(v, np.ndarray) else v.item() if isinstance(v, np.generic) else v)
                          for k, v in f.attrs.items()}
            self.times, self.frame_ids = f['timestamps'][:], f['frame_ids'][:]
            self.angles = {s: f[s+'_angles'][:] for s in SEGMENTS}
            self.valid = {s: f[s+'_valid'][:].astype(bool) for s in SEGMENTS}
        self.source_path = Path(source_path or self.attrs['input_file']).resolve()
        self.checkpoint_path = Path(self.attrs['checkpoint'])
        with h5py.File(self.source_path, 'r') as f:
            assert f.attrs['coordinate_alignment'] == self.attrs['coordinate_alignment']
            assert np.array_equal(f['frame_ids'][:], self.frame_ids)
            assert np.array_equal(f['timestamps'][:], self.times)
            self.points = {s: f[s+'_hand_keypoints'][:] for s in SEGMENTS}
        assert np.isfinite(self.times).all() and np.all(np.diff(self.times) > 0)
        self.indices, self.duration, self.xy, self.fk, self.bounds = {}, {}, {}, {}, {}
        self.hand_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
        for base in (5, 10, 15, 20):
            self.hand_edges += [(0, base)]+[(i, i+1) for i in range(base, base+4)]
        lookup = {n: i for i, n in enumerate(JOINT_NAMES)}
        self.robot_edges = [(lookup[a], lookup[b]) for a, b in JOINT_EDGES]
        for side, (start, end) in SEGMENTS.items():
            assert self.valid[side][start:end].all(), f'{side} segment contains invalid frames'
            assert np.isfinite(self.angles[side][start:end]).all()
            duration = float(self.times[end-1]-self.times[start])
            self.duration[side] = duration
            target_times = self.times[start]+np.arange(math.ceil(duration*FPS))/FPS
            # Timestamp lookup uses the most recent available original frame, not interpolation.
            idx = np.searchsorted(self.times, target_times, side='right')-1
            self.indices[side] = np.clip(idx, start, end-1)
            basis = build_l21_reference_basis(side)
            self.xy[side] = self.points[side] @ basis
            cfg = L21.hand_kinematics_config()
            fk = create_hand_kinematics(L21.left_urdf if side == 'left' else L21.right_urdf,
                                      cfg, 'cpu', scale_factor=L21.training.robot_scale)
            nodes = np.pad(self.angles[side][start:end], ((0, 0), (0, 5)))
            with torch.no_grad():
                self.fk[side] = fk.forward(torch.from_numpy(nodes))[2].numpy() @ basis
            self.bounds[side] = []
            for points in (self.xy[side][start:end], self.fk[side]):
                low, high = np.min(points, axis=(0, 1)), np.max(points, axis=(0, 1))
                pad = (high-low)*.08
                self.bounds[side].append((low-pad, high+pad))

    def comparison(self, side, index, scene, size=(1280, 720)):
        start, _ = SEGMENTS[side]
        title = ('左手' if side == 'left' else '右手')+'：输入 → 预测 → 原生手映射'
        im = page(title, '真实记录片段 · 原始输出角度 · 按时间戳采样 · 15 fps', size)
        panel_w = (size[0]-140)//3
        labels = ['人手输入 / Hand25', '模型预测 / L21 FK', 'H1-2 / 原生手目标姿态']
        clipped = scene.pose(side, self.angles[side][index])
        views = [skeleton(self.xy[side][index], self.hand_edges, self.bounds[side][0], panel_w-24, 320),
                 skeleton(self.fk[side][index-start], self.robot_edges, self.bounds[side][1], panel_w-24, 320),
                 scene.render(panel_w-24, 320, side)]
        for k, (label, view) in enumerate(zip(labels, views)):
            x = 50+k*(panel_w+20)
            card(im, (x, 186, x+panel_w, 562))
            text(im, (x+14, 200), label, 20, INK, True)
            im.paste(view, (x+12, 238))
        text(im, (50, 583), f'原始帧 {index}  |  记录时间 {self.times[index]-self.times[0]:.2f} s  |  本帧有效  |  H1 限位裁剪 {clipped} 个关节', 19, BLUE)
        text(im, (50, 621), '各面板独立显示尺度；映射为目标姿态。模型精度与物理执行效果待评估。', 18, MUTED)
        return im

    def triple(self, side, index, scenes):
        label = '左手' if side == 'left' else '右手'
        im = page(f'同一份 {label}角度，映射至三种原装灵巧手',
                  '关节语义映射 / 原生限位裁剪 · 全机缩略图 + 手部特写', (1280, 720))
        for k, (robot, scene) in enumerate(scenes.items()):
            x = 50+k*400
            clipped = scene.pose(side, self.angles[side][index])
            card(im, (x, 184, x+380, 567))
            text(im, (x+16, 198), f'{ROBOT_NAMES[robot]}   {len(scene.mapping[side])}/17 DOF', 24, INK, True)
            hand = scene.render(268, 280, side)
            im.paste(hand, (x+12, 239))
            body = scene.render(88, 150)
            im.paste(body, (x+280, 242))
            ImageDraw.Draw(im).rectangle((x+279, 241, x+368, 392), outline='#C9D6E6', width=2)
            text(im,(x+291,405),'整机',17,MUTED)
            text(im, (x+18, 531), f'限位裁剪 {clipped} 个关节', 17, MUTED)
        text(im, (50, 583), f'原始帧 {index}  |  记录时间 {self.times[index]-self.times[0]:.2f} s  |  {label}片段有效率 100%', 20, BLUE)
        text(im, (50, 621), '手臂与其他身体关节保持原姿态；原生手映射有损。动画展示目标姿态。', 18, MUTED)
        return im


def robot_images(data, scenes):
    cid = p.connect(p.DIRECT)
    try:
        bodies = []
        for robot, offset in zip(ROBOT_NAMES, (-1.8, 0, 1.8)):
            rid = RobotLoader(cid).load_robot(robot)['robot']
            pos, orn = p.getBasePositionAndOrientation(rid, physicsClientId=cid)
            p.resetBasePositionAndOrientation(rid, [offset, pos[1], pos[2]], orn, physicsClientId=cid)
            ranges, names = read_joint_ranges(rid, cid)
            mappings = {s: build_mapping(robot, s, ranges) for s in SEGMENTS}
            style_hands(p, rid, cid, {s: (m, None, None, names) for s, m in mappings.items()})
            bodies.append(rid)
        boxes = [p.getAABB(rid, i, physicsClientId=cid) for rid in bodies for i in range(-1, p.getNumJoints(rid, physicsClientId=cid))]
        lo, hi = np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0)
        distance = max((hi[0]-lo[0])/(2*math.tan(math.radians(21))*2), (hi[2]-lo[2])/(2*math.tan(math.radians(21))))*1.22
        im = page('三种人形机器人，统一仿真入口', 'H1-2 / GR1-T2 / G1 · 模型资产来自 TeleOpBench · 手臂为中性姿态')
        card(im, (50, 183, 1550, 806))
        im.paste(render_camera(cid, (lo+hi)/2, distance, 0, -8, 1490, 612), (55, 188))
        for k, robot in enumerate(ROBOT_NAMES):
            text(im, (180+k*490, 823), ROBOT_NAMES[robot], 29, INK, True)
            text(im, (180+k*490, 868), f'动作空间 {dict(h1_2=38, gr1_t2=36, g1=28)[robot]} 维', 22, MUTED)
        save_image(im, '01_three_robots.png')
    finally:
        p.disconnect(cid)
    im = page('原装灵巧手：结构不同，映射入口一致', '蓝色：左手 · 橙色：右手 · 每台机器人使用自己的原装手')
    for k, (robot, scene) in enumerate(scenes.items()):
        x = 50+k*510
        card(im, (x, 186, x+480, 900))
        text(im, (x+24, 205), ROBOT_NAMES[robot], 30, INK, True)
        for j, side in enumerate(SEGMENTS):
            scene.reset()
            scene.pose(side, data.angles[side][SEGMENTS[side][0]])
            im.paste(scene.render(448, 268, side), (x+16, 265+j*303))
            text(im, (x+24, 534+j*303), ('左手' if side=='left' else '右手')+f' / {len(scene.mapping[side])} 个映射自由度', 21, MUTED)
        scene.reset()
    save_image(im, '02_native_hands.png')


def diagrams():
    im = page('当前系统链路：可运行模块与待整合连接', '实线：已具备链路 · 虚线：仍需整合或验收 · 实时输入不等同于整机闭环')
    boxes = [(55, 245, 370, 385), (465, 245, 800, 385), (910, 245, 1240, 385)]
    node(im, boxes[0], '已采集手部 H5', '离线真实记录\nHand25 + 时间戳')
    node(im, boxes[1], '统一手部链路', '对齐 / 模型 / 18D 导出')
    node(im, boxes[2], '原生手映射回放', 'H1 / GR1 / G1\n目标姿态可视化')
    arrow(im, (370, 315), (465, 315)); arrow(im, (800, 315), (910, 315))
    node(im, (55, 490, 370, 650), '真实设备输入', '相机 / VR / 手套\n设备验收待完成', True)
    node(im, (465, 490, 800, 650), '双臂与整机命令', 'FK / IK 模块已有\n同步与映射待整合', True)
    node(im, (910, 490, 1240, 650), '任务仿真与记录', '框架已实现\n闭环与长程记录待验收', True)
    arrow(im, (370, 570), (465, 570), True, ORANGE)
    arrow(im, (800, 570), (910, 570), True, ORANGE)
    arrow(im, (630, 385), (630, 490), True, ORANGE)
    node(im, (1300, 490, 1545, 650), '评估', '真实精度\n任务成功率', True)
    arrow(im, (1240, 570), (1300, 570), True, ORANGE)
    wrapped(im, (65, 750), '阶段成果：离线手部链路已具备工程验证；整机实时控制、同步记录与任务评估是下一阶段连接重点。', 1440, 30, INK)
    save_image(im, '03_system_flow.png')

    im = page('手部重定向：数据、模型与消费端统一', 'MediaPipe21 → Hand25 → palm-local → 三帧模型 → L21 角度 → 原生手')
    labels = [('21 点输入', '每侧关键点与时间戳'), ('Hand25 + 对齐', '左右独立掌面坐标系'), ('连续三帧窗口', '输入 [B,3,25,3]')]
    for k, (label, detail) in enumerate(labels):
        x = 70+k*510
        node(im, (x, 228, x+420, 370), label, detail)
        if k<2: arrow(im, (x+420, 300), (x+510, 300))
    arrow(im, (1300, 370), (1300, 475))
    lower = [('原生手映射', '17 DOF → 12 / 11 / 7'), ('18D 输出', '固定占位 + 17 个真实 DOF'), ('PoseTransformer', '共享手部模型 + FK / loss')]
    for k, (label, detail) in enumerate(lower):
        x=70+k*510; node(im, (x, 475, x+420, 617), label, detail)
        if k>0: arrow(im, (x, 546), (x-90, 546))
    card(im, (70, 710, 1510, 891))
    text(im, (100, 735), '坐标契约贯穿数据、训练、权重与推理', 31, TEAL, True)
    wrapped(im, (100, 791), 'H5 = Dataset = training = checkpoint = inference alignment；同模式运行，缺失或交叉模式明确拒绝。失手时保留无效标记。', 1370, 26)
    save_image(im, '04_hand_pipeline.png')

    im = page('30 个任务定义，四级递进', '任务框架用于后续实验组织；尚未完成全部任务的物理条件与成功率验收')
    levels = [('01', '基础拾取放置', 5, 'pushcube / pickcube / pickplacecube'), ('02', '工具操作', 11, 'rotatefaucet / opendrawer / liftmug'),
              ('03', '双手协作', 9, 'ballbimanual / potbimanual / stackboxes'), ('04', '长时域序列', 5, 'pottomatoplate / drawerbook / twistbottlecaps')]
    for k, (number, title, count, examples) in enumerate(levels):
        x, y = 60+(k%2)*770, 208+(k//2)*325
        card(im, (x, y, x+730, y+287))
        text(im, (x+24,y+25), 'LEVEL '+number, 24, BLUE, True)
        text(im, (x+24,y+73), title, 33, INK, True)
        text(im, (x+590,y+57), str(count), 62, TEAL, True)
        text(im, (x+596,y+139), '个定义', 20, MUTED)
        wrapped(im, (x+24,y+188), examples, 675, 23)
    save_image(im, '05_task_levels.png')


def task_image():
    im = page('四级代表任务：真实代码构建的场景', '仅展示 reset 后的场景与任务对象；不表示机器人已完成任务')
    examples = [('pushcube', 'L1 推方块'), ('rotatefaucet', 'L2 旋转水龙头'), ('potbimanual', 'L3 双手抬锅'), ('pottomatoplate', 'L4 开盖并放置番茄')]
    for k, (task, title) in enumerate(examples):
        env = SimulationEnv(robot_type='h1_2', task_name=task, render=False, record=False)
        try:
            env.reset(randomize=False)
            x,y=50+(k%2)*770, 190+(k//2)*357
            card(im, (x,y,x+730,y+336))
            text(im, (x+20,y+12), title, 27, INK, True)
            im.paste(render_camera(env.client, [.47,.1,.46], 1.12, 48, -36, 690, 230), (x+20,y+60))
            text(im, (x+20,y+298), task+' · 初始场景 / 未执行任务', 20, MUTED)
        finally:
            env.close()
    save_image(im, '06_task_scenes.png')


def comparison_images(data, scenes):
    for number, side in ((7,'left'),(8,'right')):
        scenes['h1_2'].reset()
        start, end = SEGMENTS[side]
        im = page(('左手' if side=='left' else '右手')+'真实记录：输入与目标姿态对比', '同帧对应 · 面板独立显示尺度 · 保留预测差异，供后续质量诊断')
        for row, index in enumerate((start, end-1)):
            y=203+row*357
            text(im, (62,y-30), f'原始帧 {index} / 记录时间 {data.times[index]-data.times[0]:.2f} s', 20, BLUE)
            scenes['h1_2'].pose(side,data.angles[side][index])
            views=[skeleton(data.xy[side][index],data.hand_edges,data.bounds[side][0],456,265),
                   skeleton(data.fk[side][index-start],data.robot_edges,data.bounds[side][1],456,265),
                   scenes['h1_2'].render(456,265,side)]
            labels=['人手输入 / Hand25','模型预测 / L21 FK','H1-2 / 原生手目标姿态']
            for k,(view,label) in enumerate(zip(views,labels)):
                x=60+k*501
                card(im,(x,y,x+480,y+321))
                text(im,(x+12,y+9),label,22,INK,True)
                im.paste(view,(x+12,y+48))
        save_image(im, f'{number:02d}_{side}_comparison.png')


def charts():
    im=page('原生手映射覆盖率：可表达自由度 / 17', '覆盖率衡量目标手的表达范围，不是重定向精度或任务成功率')
    for k,(name,count,color) in enumerate([('H1-2',12,BLUE),('GR1-T2',11,TEAL),('G1',7,ORANGE)]):
        y=250+k*182
        text(im,(85,y),name,34,INK,True)
        d=ImageDraw.Draw(im)
        d.rounded_rectangle((360,y,1355,y+75),radius=12,fill='#DEE7F2')
        d.rounded_rectangle((360,y,360+995*count/17,y+75),radius=12,fill=color)
        text(im,(1380,y+12),f'{round(count/17*100)}%',32,color,True)
        text(im,(360,y+90),f'保留 {count}/17 个自由度；丢弃 {17-count} 个',25,MUTED)
    wrapped(im,(85,844),'默认按关节语义、符号与原生限位映射；保留原装手，不用缩放角度放大运动。',1400,27)
    save_image(im,'09_dof_coverage.png')

    evidence=json.loads((ROOT/'outputs/cuda_checks/export_comparison.json').read_text())
    im=page('工程验证：可运行、一致性与数据有效性', '数据来自已有 PR1.5 与 CUDA 验收日志；本图不是算法精度评估')
    card(im,(55,200,560,885)); text(im,(85,233),'自动回归',28,BLUE,True)
    text(im,(85,303),'180',100,TEAL,True);text(im,(320,352),'项通过',29,INK)
    text(im,(85,451),'181 项总计 / 1 项跳过',29,INK,True)
    wrapped(im,(85,514),'0 失败、0 错误。跳过项依赖项目本地缺失的历史 checkpoint。已有完整测试日志，未在本轮重新运行。',440,25)
    card(im,(585,200,1085,885));text(im,(615,233),'真实记录导出',28,BLUE,True)
    text(im,(615,303),'6515',86,TEAL,True);text(im,(925,352),'帧',29,INK)
    for y,side,label,color in [(479,'left','左手',BLUE),(596,'right','右手',ORANGE)]:
        valid=evidence[side]['valid']
        text(im,(615,y),f'{label}有效 {valid} / 6515',26,INK,True)
        ImageDraw.Draw(im).rectangle((615,y+51,1045,y+79),fill='#E2EAF4')
        ImageDraw.Draw(im).rectangle((615,y+51,615+430*valid/6515,y+79),fill=color)
    text(im,(615,769),'非有限值 0 / L21 越限值 0',23,MUTED)
    card(im,(1110,200,1545,885));text(im,(1136,233),'CPU / GPU 对比',27,BLUE,True)
    text(im,(1136,320),'最大角度差异',25,INK,True)
    text(im,(1136,391),'6.74e-6',48,TEAL,True);text(im,(1136,451),'rad（左右两侧最大值）',21,MUTED)
    text(im,(1136,562),'左 5.6624e-6 rad',22,INK)
    text(im,(1136,611),'右 6.7353e-6 rad',22,INK)
    wrapped(im,(1136,710),'证明数值一致性。完整训练收敛、实时延迟与真实动作精度待验收。',380,23)
    save_image(im,'10_validation.png')


def encode_video(name, frames):
    path=VIDEOS/(name+'.mp4')
    iterator=iter(frames)
    first=next(iterator)
    width,height=first.size
    ffmpeg=shutil.which('ffmpeg')
    process=None
    if ffmpeg:
        probe=subprocess.run([ffmpeg,'-hide_banner','-encoders'],capture_output=True,text=True)
        if 'libx264' in probe.stdout:
            process=subprocess.Popen([ffmpeg,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
                '-r',str(FPS),'-i','-','-an','-c:v','libx264','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    writer=None; codec='H.264 / libx264' if process else None
    encoder_backend='ffmpeg executable / libx264' if process else None
    if process is None:
        for code in ('avc1','mp4v'):
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*code),FPS,(width,height))
            if writer.isOpened():
                codec='H.264 / avc1' if code=='avc1' else 'MPEG-4 Part 2 / mp4v'
                encoder_backend=writer.getBackendName()
                break
            writer.release()
        if not writer or not writer.isOpened():
            raise RuntimeError('No usable MP4 encoder; do not deliver an empty video')
    gif_frames=[]; count=0
    def write(im):
        nonlocal count
        rgb=np.asarray(im.convert('RGB'))
        if process: process.stdin.write(rgb.tobytes())
        else: writer.write(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        if count%3==0:
            gif_frames.append(im.resize((640,360),Image.Resampling.LANCZOS).convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        if count==0: im.save(VIDEOS/(name+'_poster.jpg'),quality=94)
        count+=1
        if count%50==0: print('VIDEO',name,count,flush=True)
    try:
        write(first)
        for im in iterator: write(im)
    finally:
        if writer: writer.release()
        if process:
            process.stdin.close()
            if process.wait()!=0: raise RuntimeError('ffmpeg failed')
    gif_frames[0].save(VIDEOS/(name+'.gif'),save_all=True,append_images=gif_frames[1:],duration=200,loop=0,optimize=False,disposal=2)
    result={'file':path.name,'codec':codec,'encoder_backend':encoder_backend,'fps':FPS,'frames':count,'duration_seconds':count/FPS,
            'width':width,'height':height,'gif_fps':5,'silent':True}
    print('VIDEO_DONE',result,flush=True)
    return result


def video_check(path):
    cap=cv2.VideoCapture(str(path))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
    fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
    decoded_codec=bytes((fourcc >> (8*i)) & 255 for i in range(4)).decode('ascii',errors='replace')
    assert cap.isOpened() and count>=145 and 14.9<fps<15.1
    samples=[]
    for index in (0,count//2,count-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok,frame=cap.read()
        assert ok and frame.shape[:2]==(720,1280), f'Cannot decode {path} frame {index}'
        samples.append(Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)))
    cap.release()
    # Decode all frames once to ensure the complete stream is readable.
    cap=cv2.VideoCapture(str(path)); decoded=0
    while True:
        ok,_=cap.read()
        if not ok: break
        decoded+=1
    cap.release(); assert decoded==count
    sheet=Image.new('RGB',(1280,720*3),'white')
    for k,sample in enumerate(samples): sheet.paste(sample,(0,k*720))
    qa=OUT/'qa'; qa.mkdir(exist_ok=True)
    sheet.save(qa/(path.stem+'_contact.jpg'),quality=90)
    return {'file':path.name,'decoded_frames':decoded,'fps':fps,'duration_seconds':count/fps,
            'decoded_codec':decoded_codec,'begin_middle_end_ok':True}


def evidence():
    logs={'test_suite':'outputs/pr15_checks/final_suite.log',
          'export_comparison':'outputs/cuda_checks/export_comparison.json',
          'recording_footprint':'outputs/pr15_checks/recording_footprint.json'}
    result={'basis':'Existing validation artifacts; this build does not rerun the full suite',
            'artifacts':{key:{'repository_relative_source':value,'sha256':sha(ROOT/value)} for key,value in logs.items()}}
    suite=(ROOT/logs['test_suite']).read_text(encoding='utf-8')
    assert 'Ran 181 tests' in suite and 'OK (skipped=1)' in suite
    result['tests']={'total':181,'passed':180,'skipped':1,'failed':0,'errors':0,
                     'skip_reason':'local palm_local_v2 checkpoint unavailable'}
    result['cpu_gpu_comparison']=json.loads((ROOT/logs['export_comparison']).read_text())
    result['recording_footprint']=json.loads((ROOT/logs['recording_footprint']).read_text())
    return result


def main():
    global FONT_FILE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['images','videos','all','verify','cameras'],default='all')
    ap.add_argument('--angles',type=Path,default=ROOT/'outputs/hand_retargeting/palm_angles.h5')
    ap.add_argument('--source',type=Path,help='Override aligned keypoints H5 path from angle metadata')
    ap.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msyh.ttc'))
    args=ap.parse_args(); FONT_FILE=args.font
    IMAGES.mkdir(parents=True,exist_ok=True); VIDEOS.mkdir(parents=True,exist_ok=True)
    if args.phase=='verify':
        print(json.dumps([video_check(path) for path in sorted(VIDEOS.glob('*.mp4'))],indent=2))
        return
    torch.set_num_threads(1)
    data=Data(args.angles,args.source)
    inputs={str(path):sha(path) for path in [data.angle_path,data.source_path,data.checkpoint_path]}
    manifest_path=OUT/'素材元数据.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.is_file() else {}
    manifest.update({'generated_for':'2026-10-07','repository_head_at_start':'c57c165',
                     'inputs':[{'original_path':path,'sha256':digest} for path,digest in inputs.items()],
                     'alignment':data.attrs['coordinate_alignment'],'angle_metadata':data.attrs,
                     'rendering':'PyBullet DIRECT / ER_TINY_RENDERER / resetJointState targets, no physics stepping',
                     'mapping':'NativeHandAdapter(limit_mode=clamp); explicit 18D to 17DOF; no repair, rescale, interpolation or model retraining',
                     'sampling':'15 fps; latest original frame at each output timestamp; original angles unchanged',
                     'versions':{name:importlib.metadata.version(name) for name in ['torch','pybullet','numpy','h5py','Pillow','opencv-contrib-python']},
                     'segments':{side:{'start_frame_inclusive':a,'end_frame_exclusive':b,
                       'source_duration_seconds':data.duration[side], 'valid_frames':int(data.valid[side][a:b].sum()),
                       'output_frames':len(data.indices[side]),'selected_source_indices':data.indices[side].tolist(),
                       'source_start_seconds':float(data.times[a]-data.times[0])} for side,(a,b) in SEGMENTS.items()},
                     'evidence':evidence()})
    scenes={}
    try:
        for robot in ROBOT_NAMES:
            scene=scenes[robot]=Scene(robot)
            for side,(a,b) in SEGMENTS.items(): scene.prepare_hand(side,data.angles[side][a:b])
        if args.phase=='cameras':
            qa=OUT/'qa';qa.mkdir(exist_ok=True)
            for side in SEGMENTS:
                im=Image.new('RGB',(1600,1200),'white')
                for row,(robot,scene) in enumerate(scenes.items()):
                    scene.reset();scene.pose(side,data.angles[side][SEGMENTS[side][0]])
                    center,distance,_,_=scene.cameras[side]
                    for col,yaw in enumerate([0,45,90,135,180,225,270,315]):
                        view=render_camera(scene.cid,center,distance*.75,yaw,-12,200,350)
                        im.paste(view,(col*200,row*400))
                        text(im,(col*200+5,row*400+351),f'{robot} / {yaw}',17)
                im.save(qa/(side+'_camera_angles.jpg'))
        if args.phase in ('images','all'):
            robot_images(data,scenes); diagrams(); task_image(); comparison_images(data,scenes); charts()
        if args.phase in ('videos','all'):
            manifest['videos']=[]
            for side in SEGMENTS:
                for scene in scenes.values(): scene.reset()
                frames=(data.comparison(side,int(i),scenes['h1_2']) for i in data.indices[side])
                manifest['videos'].append(encode_video(side+'_comparison',frames))
                for scene in scenes.values(): scene.reset()
                frames=(data.triple(side,int(i),scenes) for i in data.indices[side])
                manifest['videos'].append(encode_video('three_robots_'+side,frames))
            manifest['video_validation']=[video_check(path) for path in sorted(VIDEOS.glob('*.mp4'))]
    finally:
        for scene in scenes.values(): scene.close()
    assert inputs=={str(path):sha(path) for path in [data.angle_path,data.source_path,data.checkpoint_path]}, 'Protected inputs changed'
    manifest['protected_input_hashes_unchanged']=True
    manifest['images']=[{'file':f.name,'size':list(Image.open(f).size),'sha256':sha(f)} for f in sorted(IMAGES.glob('*.png'))]
    manifest['outputs']=[{'relative_path':str(f.relative_to(OUT)).replace('\\','/'),'bytes':f.stat().st_size,'sha256':sha(f)}
                         for folder in (IMAGES,VIDEOS) for f in sorted(folder.iterdir()) if f.is_file()]
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE; protected input hashes unchanged',flush=True)


if __name__=='__main__':
    main()

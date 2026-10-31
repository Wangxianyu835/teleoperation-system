# 手部重定向链路怎么跑、怎么复核（current-state 布局）

> 手部训练/控制/导出接口已更新，当前可执行命令见 [统一手部流程](hand-retargeting-unified.md)。本页原有共享18D训练及输出段落是历史记录，不能用于新版模型；采集仍保存米制人体数据。

> 2026-10-11：本文保留原分支的运行与验收历史。合并后的入口、坐标数据差异及测试结果见 [main 集成记录](integration-2026-10-11.md)，本文的历史机器路径与测试数量不代表当前 main。

> **适用版本**：`origin/current-state`（包 `src/teleoperation`，CLI `python -m teleoperation`）
> [`contracts.md`](contracts.md)（H5 / 命令契约，字段以它为准）· [`architecture.md`](architecture.md)（模块层次）·
> 注意：`docs/archive/`（旧布局手册，只读参考）已随上游 `10ebd9f` 被整体删除，
> 旧布局的对照表保留在 §5（本文档内，自足）。
>
> 本文只讲**怎么用现有代码跑通并复核**；算法/契约本身由队友维护，**不要为跑通而改 `src/teleoperation`**（CONVENTIONS 约定 3）。

---

## 0. 一次性准备

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"                                  # 本机没有 conda 时必须
E:\python3.11.7\python.exe -m teleoperation --help             # 期望退出码 0
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1  # 环境自检，期望 RESULT [OK]
```

顶层子命令（`--help` 实测 2026-10-10）：

| 子命令 | 管什么 |
|---|---|
| `hand {align,train,export,inspect,realtime}` | 手部角度对齐 / 训练 / 导出 / 检查 / 实时（另有一个**不进统一 CLI** 的实时可视化入口，见 §4.2）|
| `replay {actions,native-hand,l21-hand,mounted-hand}` | 回放：动作序列 / 原生手 / l21 手 / 装到机器人手上 |
| `sim {run,demo-joints}` | 任务仿真（`pushcube`/`pickcube` 等）与关节演示 |
| `dual {export,realtime,replay}` | 双臂（离线导出 / 实时 / 回放）|
| `tools {...}` | 一堆自检/可视化工具（见 §4）|

---

## 1. 离线链路：原始关键点 -> 手部角度 -> 关节命令

```powershell
$py='E:\python3.11.7\python.exe'

# (1) 原始关键点 H5 -> 归一化手部角度 H5（对齐左/右，每侧 18 维）
& $py -m teleoperation hand align --input  datasets\raw\retarget_twohand_153542.h5 `
                                   --output outputs\tmp_hand\angles.h5

# (2) 复核角度文件（帧数 / 维度 / 有效帧 / 越限）
& $py -m teleoperation hand inspect --angle-h5 outputs\tmp_hand\angles.h5

# (3) 训练（可选；不训练就用已有 checkpoint 或 hand align 的直出结果）
& $py -m teleoperation hand train --input outputs\tmp_hand\angles.h5 --run-name my_run --device cpu

# (4) 用 checkpoint 导出角度（把模型输出写回 H5）
& $py -m teleoperation hand export --input datasets\raw\retarget_twohand_153542.h5 `
                                   --checkpoint <checkpoint 路径>.pt `
                                   --output outputs\tmp_hand\angles_from_ckpt.h5 --device cpu
```

- `hand align` 只有两个参数：`--input` / `--output`（实测）。
- `hand export` 还有 `--batch-size` / `--scale-factor` / `--disable-identity-tracking`
  / `--max-center-displacement` / `--max-shape-rmse` / `--device {auto,cpu,cuda}`。
- 产出文件一律写 `outputs\...`（`.gitignore` 已忽略；CONVENTIONS 约定 7）。
- **H5 字段与语义以 [`contracts.md`](contracts.md) 为准**，本文不重复定义。

## 2. 回放与可视化（每步都能单独验）

```powershell
$py='E:\python3.11.7\python.exe'

# l21 手：角度 -> l21 关节（可选 --check 做检查、--render 开 GUI、--analyze 出统计）
& $py -m teleoperation replay l21-hand --file outputs\tmp_hand\angles.h5 --check

# 原生手：把角度直接喂给机器人自带手（--robot h1_2/gr1_t2/g1，可 --render --view left-hand）
& $py -m teleoperation replay native-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 把手装到机器人手腕上（检查安装位姿/抖动）
& $py -m teleoperation replay mounted-hand --file outputs\tmp_hand\angles.h5 --robot h1_2 --render

# 动作序列（契约 H 的 .npz/.h5）或假数据，无头跑，最省事的冒烟
& $py -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render
& $py -m teleoperation replay actions --file datasets\...\actions.npz --robot h1_2 --no-render
```

## 3. 任务与整机

```powershell
$py='E:\python3.11.7\python.exe'
& $py -m teleoperation sim run --task pushcube --robot h1_2 --no-render --demo    # 有头去掉 --no-render
& $py -m teleoperation sim demo-joints                                            # GUI 里看关节
& $py -m teleoperation dual export --observations <obs> --angle-h5 <angles.h5> --calibration <file> --output outputs\tmp_dual\cmd.h5
& $py -m teleoperation dual replay --command-h5 outputs\tmp_dual\cmd.h5 --calibration <file>
```

`sim run` 的动作空间实测为 **38 维** = 左臂 7 + 右臂 7 + 左手 12 + 右手 12（`tools check-environment` 会打印）。

---

## 4. 一键复核（改动前后都要跑同一条）

| 命令 | 判据 | 说明 |
|---|---|---|
| `python -m teleoperation tools verify-hand-pipeline` | 退出码 `0`，末行 `结论：... [PASS]` | **整条离线链路的验收器**：读真实 H5（557 帧 × 18 维）→ 逐项检查 → 仿真回放。旧布局的 `scripts/run_retargeting_pipeline.py` 对应物 |
| `python -m teleoperation tools check-environment` | 退出码 `0` | 项目自带环境自检：6 项，含 300 步无头端到端（`[OK] 端到端跑通（robot_id=1, action_dim=38）`）|
| `python -m unittest discover -s tests -v` | 见 [`CONVENTIONS.md`](CONVENTIONS.md) 约定 5 / §3.2 | 本机回落下实测 `Ran 19 tests / failures=0 / errors=0 / skipped=1`（2026-10-10 上游 `10ebd9f` **删除了整个 `tests/`（32 文件）**，所以现在只有本支的 2 个文件、19 个用例；`PYTHONUTF8=1` 仍建议带上）；判据是**不劣化** |
| `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full` | 退出码 `0` | 上面三条 + 4 条冒烟，共 8 项一次跑完（含 GBK 两条、架构边界、C 盘零写）|

`tools` 里还有一批单项工具（`--help` 实测）：`validate-retarget-input`、`check-gbk-safe`、
`check-camera`、`show-robots`、`show-hand`、`show-all-hands`、`visualize-hand-keypoints`、
`visualize-l21-fk`、`compare-training-hand-pose`、`diagnose-hand-coordinates`、`make-sample-data`、`verify-p0`。
排查问题时优先用它们，**不要**先改代码。
### 4.1 几何重定向后端（不依赖 checkpoint 的备用路线）

`src/teleoperation/retargeting/hand/geometric.py` 的 `GeometricHandRetargeter` 是实时循环在
不加载 checkpoint 时用的后端（MediaPipe 21 点 -> 掌面局部对齐 -> 按**关节自身轴**测角）。
它的正确性由这条**往返实测**把关 —— 与上面的表格一起构成"改前基线 / 改后复验"：

```powershell
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe devtools\verify_geometric_hand.py   # 期望退出码 0，末行 [OK] 全部判据通过
```

- 做法：用仓库自带 L21 URDF + FK 造出**已知角度**的手 -> 按采集格式反推成 21 点输入 ->
  走 `21 点 -> ensure_hand25 -> 掌面局部对齐 -> 三帧窗口 -> 几何重定向` -> 与真值逐维比对。
- 判据（左右手各 13 组姿态）：真值为 0 的维 `|err| <= 0.05`；真值非 0 的维 `value >= 0.10` 且 `|err| <= 0.25`。
- 实测 2026-10-10：四指全部 `max_err <= 0.001`；拇指对掌姿态曾 **4 条判据不过**
  （左手 `dim14 0.700 -> +0.354`、`dim15 0.400 -> +0.145`）。改成**拇指串链反解**
  （对 cmc_yaw/cmc_pitch 做 FK 反解，mcp/ip 再用父链旋转后的轴测角）后全部通过：
  左手 `0.693 / 0.397 / 0.803 / 0.900`，右手 `0.709 / 0.374 / 0.820 / 0.900`，
  各维最大误差 `d14 0.009 / d15 0.026 / d16 0.020 / d17 0.000`。
- 成本实测（`E:\python3.11.7`，单线程）：单次链式求解 `0.92 ms`，两只手一帧 `1.83 ms`，
  整条 `retarget` `3.13 ms/帧`（30 fps 的预算是 33 ms/帧）。
- 改这个后端后，除了跑本命令，还要跑 `tools verify-hand-pipeline`（真实 H5 的端到端验收）。



### 4.2 实时相机机械臂与三机器人仿真

当前相机身体输入、模型准备、录制、实时控制和身体 H5 回放见 [SMPLer-X 链路](smplerx-arm.md)。
机械臂映射、腕控制与 IK 分层见 [多机器人机械臂控制](multi-robot-arm.md)。
手部独立训练和控制使用 [统一手部流程](hand-retargeting-unified.md)。

### 4.3 没有摄像头也能出画面：文件驱动的同一套链路（`devtools/replay_capture_to_l21.py`）

实时入口的输入是摄像头，所以"镜头前有没有手"直接决定它能不能显示手。2026-10-10 在本机
（非交互会话、镜头前没有手）做过一组取证，结论是**代码没问题，是这一次输入里没有手**：

| 证据 | 怎么取的 | 结果 |
|---|---|---|
| 相机本身正常 | `outputs\tmp_probe\camera_brightness_probe.py` | index 0 可打开，20/20 帧，640x480，亮度 114~147（不是黑屏、不是被遮挡） |
| 队友自己的采集脚本 | `hand capture media.py` 原样复制成 `outputs\hand_capture_probe.py`，只加"跑 N 帧自动停 + 不开窗口"两个开关 | 120 帧全部 `有效手部帧数：0` |
| 同一脚本 2026-09-12 采的数据 | `visual_hand_data_20260912_112108.h5` / `_111415.h5` | 左手 643/701、右手 654/701 帧有 21 点（另一份 367/402、278/402） |
| 本仓库没有任何检查点 | `outputs/**` 与 `D:\2026\code\mytrans` 下 `*.pth` 数量 | 0 个 -> 只能用几何后端（零权重） |

同一份代码在"镜头里有手"时能稳定检出（9-12 的数据就是证据），本次 0 帧只是因为镜头前没有手；
实时入口打印的 `valid=left=0 right=0` 说的就是这件事。2026-10-10 加了摄像头预览窗口后再取一次证，
结论更具体：预览 PNG 里画面是**天花板斜上方**（墙面、挂钟、头顶头发），30 帧 `raw=0/30`；
12 帧那次偶然出现的骨架落在头发区域上，属于误检而不是手。也就是说根因是**取景**（镜头没对着手），
不是代码 —— 把摄像头角度调低或把手抬到镜头正前方即可，预览窗口会实时显示手在不在画面里。
为此补了一个**文件驱动**的入口，
除了输入来源不同，链路与实时入口完全一致（掌面局部对齐 -> 3 帧窗口 -> 重定向 -> L21 PyBullet），
同样不改任何队友文件：

```powershell
$py = 'E:\python3.11.7\python.exe'
# 无头渲染：每 50 帧存一组 iso/side 视图 PNG，并写 JSON 报告
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --out-dir outputs\tmp_probe\offline_l21 `
      --views iso side --render-every 50 --report outputs\tmp_probe\offline_l21\report.json
# 本地看窗口（需要交互式桌面；--hold-seconds 让最后一帧留在屏幕上）
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --gui --hold-seconds 30
# 整段视频：每帧渲染成 mp4，分辨率 = --width x --height（默认 640x480 iso 视角）
& $py devtools\replay_capture_to_l21.py --input-h5 <raw.h5> --out-dir outputs\tmp_probe\offline_l21_video `
      --render-every 0 --video replay_l21_both.mp4
# 输入已经是 18 维角度 H5：跳过 21 点 -> 掌面局部 -> 重定向，直接驱动三台机器人
& $py devtools\replay_capture_to_l21.py --angles-h5 datasets\raw\retarget_twohand_153542.h5 `
      --out-dir outputs\tmp_probe\offline_robots_twohand --render-every 0 `
      --video replay_twohand_robots.mp4 --report outputs\tmp_probe\offline_robots_twohand\report.json
```

- 输入契约：`left_hand_keypoints` / `right_hand_keypoints`，形状 `(T,21,3)`；没检出的一侧写成全 0
  （与 `retargeting/data.py` 的说明一致），本入口把全 0 帧当作"这一帧没有这只手"。
- **角度直通**（`--angles-h5`，与 `--input-h5` 二选一；两个都给会在 `main()` 里直接报错退出）：
  输入契约是 `left_angles` / `right_angles`，形状 `(T,18)`（18 = `HAND_ANGLE_DIM`）；
  可选的 `left_valid` / `right_valid` `(T,)` 标出哪一帧有这只手（没有这两个数据集时整段视为有效），
  `timestamps` / `frame_ids` 只写进报告、不参与数值。既然每帧本身已经是重定向结果，
  `21 点 -> 掌面局部 -> 三帧窗口 -> 重定向` 这**整段跳过**，所以 `--calibration-frames` /
  `--smoothing` 对这条路径**不生效**（脚本会打印一行 note 说明），报告里
  `input_schema="angles"`、`calibration="n/a (...)"`，一眼能区分这次跑的是哪条输入。
- **视频分辨率**：`--video` 写出的 mp4 尺寸 = `--width x --height`（默认 `640x480`，4:3、`iso` 视角、
  逐帧写）。320x240 下手指只有几个像素、看不出屈曲，所以要看手就用默认 640x480；代价是渲染时间随
  像素数近似线性增长（本机 320x240 约 `0.28 s/帧`、640x480 约 `0.35 s/帧`），整段跑大分辨率时可以
  `--stride 2 --fps 15`：帧数减半、播放时长不变。
- **判"手真的在动"用关节统计，不要用画面逐帧像素差**：`ER_TINY_RENDERER` 对棋盘地面有抖动噪声，
  实测连"两边都没手、姿态完全没变"的两帧（第 0 / 20 帧）也有 `mean|diff| 2.3`、`12.8%` 像素变化，
  和真有动作的帧（`2.4` / `12.9%`）分不开。可信的判据是报告里的 `angles.max_std`
  （逐维标准差的最大值 > 0 就说明姿势在变）与 `final_joint_values`。
- 脚本自带 `src` 的 `sys.path` 引导，所以按**文件路径**直接跑即可，不需要先设 `PYTHONPATH`。
- 无显示环境也能出证据：用 PyBullet 的 CPU 光栅化（`ER_TINY_RENDERER`）出 PNG / mp4，
  所以"到底有没有把手渲染出来"可以截图核对，不必依赖 GUI。
- 验证记录（2026-10-10，`--input-h5 visual_hand_data_20260912_112108.h5`，双手，701 帧）：
  - 处理 701 帧用 `12.05 s`（`58.2` 帧/秒）；左手检出 643 帧 -> 输出 627 组角度，右手检出
    654 帧 -> 输出 638 组角度；两侧开手标定均完成（`calibration={'left': True, 'right': True}`）。
  - 角度确实随数据变化（不是一张静态图）：18 维角度的最大逐维标准差 左 `0.6801 rad` /
    右 `0.6703 rad`；指节屈曲接近满量程（`index_pip` 0 -> 1.57），拇指 yaw 随握合变化（0.324 / 0.942）。
  - 出图：`--render-every 50` 得到 28 张 PNG（`outputs\tmp_probe\offline_l21\`，另附 `sheet_all.png` 拼图）；
    全帧视频 `outputs\tmp_probe\offline_l21_video\replay_l21_both.mp4`（`1,037,756` 字节，701 帧）。
  - 目视核对（截图中可见两只 L21 手，浅色是左手、深色是右手）：第 99 帧四指伸直、第 349 帧屈曲、
    第 699 帧侧视图手指收拢。
- 退出码：任一侧产出过角度 -> `0`；一帧都没产出 -> `1`。
- `devtools\replay_capture_to_l21.py` 的 `--distance` / `--target-z` 只影响取景，不影响角度数值。
- `--scene` 同样适用（默认 `robots`：三台并排 + 原装手；`--scene hands` 回到只有 L21 手）。
  `--scene robots` 时取景与裁剪面按机器人整机给（`target=[0.13, 0, 0.86]`、`distance=4.09 m`、
  fov 60 / far 26 m），并且**默认视角自动换成 `front + iso`** —— `side`（yaw 90）是"沿着排看"，
  三台会前后遮挡（实测只剩最前面一台）。
- 验证记录（2026-10-10）：本机没有那份采集 H5，用 `datasets/samples/human_hand_demo_right.h5` 的
  `keypoints_3d (100,21,3)` 临时转成 `left/right_hand_keypoints` 结构后跑 A/B：
  `--scene hands` 退出码 `0`（右手 84 组角度、`max_std 0.2213 rad`、4 张 PNG）；
  `--scene robots` 退出码 `0`（同一份角度、4 张 PNG -> `outputs\tmp_probe\offline_robots\`），
  目视核对 `frame00049_front.png` 与 `frame00099_iso.png` 都是三台机器人。
- 验证记录（2026-10-10，`--angles-h5 datasets/raw/retarget_twohand_153542.h5`，557 帧，双手，`--scene robots`）：
  - 这份 H5 是**真采集数据**推出来的角度：属性里记着
    `input_file=.../visual_hand_data_20260912_153542.h5`、
    `checkpoint=.../twohand_h5/linker/none_warmstart/model_best.pth`、`left_coordinate_mode=none`；
    数据集为 `left_angles/right_angles (557,18)`、`left_valid/right_valid (557,)`、`timestamps/frame_ids`。
  - 无头（`--render-every 0`，不出视频）退出码 `0`：557 帧 `6.68 s`（`83.3` 帧/秒）；
    左手有效 510 帧、右手有效 512 帧。有效掩码的实际分布是 `left_valid` 在 `0..24` 与 `535..556`
    为 `False`、`right_valid` 在 `0..24` 与 `537..556` 为 `False`（这些行的 18 维角度同时是全 0），
    即**有效区间约在 25..534**。
  - 角度确实随数据变化：最大逐维标准差 左 `0.7050 rad` / 右 `0.5516 rad`；`report.json` 里
    `input_schema="angles"`、`calibration="n/a (--angles-h5: the 18D angles are already retargeted)"`。
  - 整段视频两版：
    - `outputs\tmp_probe\offline_robots_twohand\replay_twohand_robots.mp4` —— `1,034,938` 字节、
      557 帧（每帧都写）、320x240、30 fps，改分辨率之前跑的那版；取帧人眼复核
      （第 0 / 150 / 300 / 450 / 556 帧）：三台都在画面内、棋盘地面正常，第 0 帧是 `valid=False`
      的静止姿态，其余帧手部姿势各不相同。
    - `outputs\tmp_probe\offline_robots_twohand_hd\replay_twohand_robots_hd.mp4` —— `1,539,534` 字节、
      279 帧（`--stride 2`）、640x480、15 fps（播放时长与上一版同为 `18.6 s`）；退出码 `0`，
      渲染 `97.52 s`（`2.9` 帧/秒）；左手有效 255 帧 / 右手 256 帧、`max_std` 左 `0.7067` /
      右 `0.5458 rad`。
    - 放大复核（把 `GR1-T2` 的 `250,195-375,275`、`H1-2` 的 `140,160-240,230` 裁出来 x5 拼成
      逐帧对照图）：灰机器人手上**手指轮廓逐帧不同**（第 15/75 帧收拢、第 225/265 帧张开成叉状），
      黑机器人的手臂/手在第 195/225 帧相比第 15 帧明显转开 —— 与关节统计一致，手确实在动。
- **踩坑记**：这份 H5 的有效区间在 `25..534`，所以 `--max-frames 12` 这种"只看开头几帧"的跑法
  会整段落在无效区而**正常返回退出码 `1`**（不是 bug）；要抽查就配 `--stride` 让它跨进有效区，
  或干脆整段跑（557 帧无头只要 `6.7 s`）。
- 互斥与报错的两条负路径（实测 2026-10-10）：同时给 `--input-h5` 和 `--angles-h5` ->
  退出码 `2`、`error: pass exactly one of --input-h5 (raw 21 landmarks) or --angles-h5 (18D angles)`；
  把 **21 点**的 H5 塞给 `--angles-h5` -> 退出码 `1`、
  `lacks datasets ['left_angles', 'right_angles']; found ['frame_ids', 'left_hand_keypoints', 'right_hand_keypoints', 'timestamps']`。
  角度直通同样支持 `--scene hands`（只用 L21 手）：同一份数据前 120 帧退出码 `0`、
  左右各 95 帧有效（`max_std` 左 `0.7346` / 右 `0.6022 rad`）。
- 改分辨率后的老路径回归（2026-10-10，21 点输入）：`--input-h5 <21 点 H5> --width 320 --height 240
  --render-every 0 --video replay_check.mp4` 退出码 `0`，日志里
  `video: ...\replay_check.mp4 320x240 @ 30 fps`，右手 `100/100` 帧检出 -> `84` 组角度、
  `max_std 0.2213 rad`（与改动前同一份数据的数字一致），说明 `--angles-h5` 分支与新的
  视频分辨率参数都没有动到老的 21 点路径。

---

### 4.4 摄像头 -> H5：离线闭环的起点（`teleoperation.applications.camera_record`，2026-10-10 新增）

**为什么需要**：`hand align` 与 `--input-h5` 回放都要求输入是"根数据集
`left_hand_keypoints` / `right_hand_keypoints`、形状 `(T,21,3)`"的**原始抓取 H5**。
队友原来的采集脚本（产物形如 `input\visual_hand_data_20260912_112108.h5`，来自
`D:\2026\code\mytrans`）没有随仓库入库、本机也已不存在，所以"摄像头到文件"这一环
一直没有入口 —— 现在补上了。

| 新增文件 | 干什么 |
|---|---|
| `src/teleoperation/data/capture_h5.py` | 抓取 H5 的容器（增量写、形状/dtype/attrs 与队友读取器逐字段对齐）|
| `src/teleoperation/applications/camera_record.py` | 入口：摄像头 -> H5（可选再落一份 18 维角度 H5）|
| `devtools/verify_camera_record.py` | 一键验收：录制 -> 读回 -> `hand align` -> `hand inspect` ->（有手时）回放 |
| `tests/test_camera_record.py` | 10 个用例，全部不需要摄像头/不需要 MediaPipe 运行时 |

它**不进统一 CLI**（`cli/registry.py` 属队友文件，约定 3），按模块路径跑：

```powershell
$py='E:\python3.11.7\python.exe'
# 一键验收（真摄像头 10 秒；无人值守会话也能跑）
& $py devtools\verify_camera_record.py --seconds 10
# 只录制：原始点 + 18 维角度（角度来自零权重几何后端，见 §4.1）
& $py -m teleoperation.applications.camera_record --output-h5 datasets\raw\my_capture.h5 `
       --angles-h5 outputs\tmp_hand\my_capture_angles.h5 --seconds 10
```

**落盘约定**（沿用队友既有语义，不新造）：未检出的那一侧写**全零行**
（`align_recording` 会统计成 `zero_source_frames`，`devtools/replay_capture_to_l21.py`
会把全零还原成"没有手"）；另写 `left_valid` / `right_valid` 让"到底有没有检出"可直读；
`frame_ids` 取自 MediaPipe `metadata.frame_index`；`timestamps` 为**相对第一帧的秒**
（与 `datasets\raw\*.h5` 同口径），`unix_ms` 存原始 Unix 毫秒；attrs 里
`source_landmark_space=mediapipe_normalized`、`missing_hand_encoding=zeros`、
`timestamp_unit=relative_seconds`。

**实测 2026-10-10**（`E:\python3.11.7\python.exe`）：

| 命令 | 结果 |
|---|---|
| `devtools\verify_camera_record.py --seconds 10` | 退出码 `0`；`frames=288`、`duration_seconds=9.513`、`measured_fps=30.274`、`loop_fps=28.785` |
| 同上，第 2 步（读回） | `capture layout is (T,21,3)+valid+frame_ids+timestamps+unix_ms` |
| 同上，第 3 步（`hand align`） | `aligned_*.h5 (T,25,3)`，`right_zero_source_frames=288`（这一段镜头前没有手，所以全是零行）|
| 同上，第 4 步（`hand inspect`） | `angle H5 frames=288 valid left=0 right=0 backend=geometric` |
| 同上，第 5 步（回放） | `[SKIP]`（设计如此：本段没有检出手，脚本明确打印 `[WARN] no hand was in frame, so this run proves the file chain only`）|
| `python -m unittest tests.test_camera_record -v` | `Ran 10 tests ... OK`（含一个"打桩摄像头跑完整 `run()`"的回归用例，就是它抓到了 `close()` 后 summary 少字段的 bug）|

结论：**"录制 -> 读回 -> 对齐 -> 角度 -> 可复核"这条文件链路已经用真摄像头端到端验过**；
"有手时的重定向/回放"仍需镜头前真的有手（人工步骤，见 §4.2 的取景说明）。


### 4.5 身体观测到机械臂目标

相机机械臂通过 SMPLer-X 身体与腕观测进入控制，不从手指关键点推算手臂目标。
当前调用与几何公式见 [多机器人机械臂控制](multi-robot-arm.md)。

## 5. 旧布局 → 新布局对照表（合并前的习惯怎么迁过来）

| 旧（`3e4e763` 及之前）| 新（`origin/current-state`）|
|---|---|
| `python -m retargeting ...` | `python -m teleoperation hand ...` |
| `scripts/run_retargeting_pipeline.py`（我写的一键流水线）| `python -m teleoperation tools verify-hand-pipeline` |
| `inspect_angle_h5.py` | `python -m teleoperation hand inspect --angle-h5 <angles.h5>` |
| `main_offline_dual_teleop.py` | `python -m teleoperation dual export` / `dual replay` |
| `scripts/replay_hand_native.py` | `python -m teleoperation replay native-hand` |
| `retargeting/**`、`input_adapters/**`、`config/**` | `src/teleoperation/**`（同一个包内，路径变了）|
| `pytest tests -q`（40 passed / 3 skipped）| `python -m unittest discover -s tests -v`（19 tests，均为本支新增；上游 `10ebd9f` 起仓库里已无 `tests/`）|
| `robots/from_teleopbench/**` | `assets/robots/**` |
| `data/**`、`outputs/**` | `datasets/**`、`outputs/**` |
| 仓库根 `.venv` 空壳、`E:\python3.11.7` 全路径解释器 | 正式环境是 conda `teleoperation`；本机仍回落 `E:\python3.11.7` + `PYTHONPATH=src` |

**迁移时踩到的两个小坑**（已写进 CONVENTIONS）：
`src/` 布局下必须带 `$env:PYTHONPATH="$pwd\src"`（包没装进回落解释器）；
`tmp_*` 在本仓库**不会**被 `.gitignore` 忽略（只有 `outputs/` 会），临时文件要放 `outputs\tmp_*\`。

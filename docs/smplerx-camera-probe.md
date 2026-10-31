# 摄像头到 SMPLer-X 输出诊断

从仓库根目录运行（输出目录必须尚不存在）：

```powershell
& D:/Anaconda/envs/teleoperation/python.exe devtools/probe_smplerx_camera.py `
  --camera-id 0 --camera-backend dshow --frames 30 --preview `
  --output-dir outputs/smplerx_probe_manual
```

去掉 `--preview` 可以无窗口运行。`--camera-backend auto/msmf` 可测试其他采集后端。
`--input-image PATH` 会重复测试已保存图片，排除现场运动因素。
四项模型资产参数与正式应用一致，见 `--help`；启动器自动使用隔离模型解释器。
退出码：0 表示至少有一帧最终双臂有效；1 表示完成测试但没有有效身体结果；2 表示异常。
`passed` 仅证明有有效输出，不证明动态精度、连续有效率或实时性能达标。

## 正常返回什么

官方流程先使用 MMDetection Faster R-CNN 检测 person，裁剪 RGB 图像，再执行
`model({'img': image}, {}, {}, 'test')`，返回张量字典。没有检测框时不会进入姿态网络。
官方程序导出姿态参数 NPZ，可选导出 OBJ 网格，并输出裁剪/相机元数据及渲染图片。
姿态网络本身没有一个统一的“人体识别成功置信度”；框分数属于前置检测器。
当前检测分数阈值为 0.3；多框直接选择，已取消歧义保持和身份三帧恢复。
具体规则见 `docs/smplerx-operator-recovery.md`。下文带日期的实测描述保留当时的旧策略结果。

来源：

- [官方 inference.py](https://github.com/SMPLCap/SMPLer-X/blob/main/main/inference.py)
- [官方 SMPLer_X.py](https://github.com/SMPLCap/SMPLer-X/blob/main/main/SMPLer_X.py)

本机 S32、单人、batch=1 的实际原始输出：

| 键 | 形状 | 含义 |
| --- | --- | --- |
| img | 1×3×512×384 | 输入网络的归一化 RGB 裁剪图 |
| joint_img | 1×65×3 | 网络关节点，内部热图坐标 |
| smplx_joint_proj | 1×137×2 | 模型关节投影，内部坐标，不能直接当原图像素 |
| smplx_mesh_cam | 1×10475×3 | 相机坐标人体网格顶点 |
| smplx_root_pose | 1×3 | 根轴角（弧度） |
| smplx_body_pose | 1×63 | 21 个身体关节轴角 |
| smplx_lhand_pose / smplx_rhand_pose | 各 1×45 | 各手 15 个关节轴角 |
| smplx_jaw_pose | 1×3 | 下颌轴角 |
| smplx_shape / smplx_expr | 各 1×10 | 体型 / 表情参数 |
| cam_trans | 1×3 | 相机平移 |
| lhand_bbox / rhand_bbox / face_bbox | 各 1×4 | 内部区域框 |

项目在原始输出外增加 `SmplerXResult`：包含 frame_id、完成时间、模型 ID/hash、
`valid.left/right`、`reason`、metadata 及 arrays。有效时 arrays 包含
`joints_root_m (55,3)`、`global_rotations (55,3,3)`、`image_joints (55,3)`，
以及去 batch 维的姿态参数、固定体型与逆裁剪矩阵等。
`image_joints` 的 x/y 是还原到原图的归一化坐标。
无效结果的 arrays 可以为空，即使该帧已实际执行网络，例如体型初始化期间。

## 保存内容与判读

每帧一个目录：`input.png` 为实际输入画面，`overlay.png` 为检测框及有效时的双臂投影，
`frame.json` 包含检测分数、过滤原因、是否执行网络、各数组形状/范围/有限性、
最终 valid/reason 与耗时。`raw_model.npz` 保留所有原始输出张量（含 batch 维及输入裁剪）；
`result.npz` 保留项目结果 arrays。未执行网络时没有 raw_model.npz。
顶层 `report.json` 汇总每帧，`worker.stderr.log` 保存模型加载警告和异常堆栈。

- `no_accepted_person`：查看 raw_count 与 rejected_by，区分没有框和被分数/尺寸过滤。
- `shape_initializing`：网络已运行，等待五个稳定体型样本，并非没有识别到人。
- `operator_rejected`：身份保护拒绝，例如多人、框跳变或采样间隔超过 2 秒。
- `input_expired`：进入推理前帧年龄已超过 0.5 秒。
- `valid_body`：项目后端已产生双侧有效身体输出。
- `failed`：查看 stage/error 和 stderr，区分资产加载、摄像头及推理异常。

探针只在独立进程对现有 backend 安装观察钩子，保持网络、权重、阈值、裁剪、体型与身份策略。
它加载模型后才采帧，避免把模型加载时间算作输入帧年龄。这里直接调用现有后端，
不测正式 worker IPC、跟踪器、IK 或机械臂。保存张量会引入同步、复制与磁盘开销，
不能将探针耗时当作正式实时性能，采样间隔也可能因此触发身份保护。

## 2026-10-31 本机实测

默认 auto 后端在受限执行环境无法打开摄像头 0。主机权限下 DirectShow 成功采集 10 帧。
第一帧检测器分数 0.989702，唯一框通过过滤；原始网络输出上述全部 16 项张量且均有限。
第一帧最终结果为体型初始化，耗时 2140 ms。前两帧采集间隔 2.25 秒，
第二帧仍有唯一人体框（分数 0.989234），但身份状态被锁定，后续不再执行姿态网络。
10 帧最终有效数为 0，网络执行数为 1。

本次证据排除了“完全没有检测到人体 / 姿态网络没有任何输出”，表明此次测试被身份策略阻断。
超过两秒的间隔包含冷启动推理与探针保存开销，原实时链路是否存在同样间隔需独立核对。
本次任务仅增加诊断，不修改正式身份保护策略。

另将第一帧现场 `input.png` 作为静态输入重复推理 7 次：网络 7 次均运行，
前 4 次为体型初始化，第 5–7 次返回 `valid.left=True, valid.right=True`。
该次首帧耗时 1719 ms，未触发 2 秒身份锁定，退出码为 0。
结果见 `outputs/smplerx_probe_saved_camera_20261031/`。
这证明摄像头实际拍到的同一幅图像可以产生完整有效身体结果，但不证明动态链路稳定。

本机上游来源版本 `064baef0e4ab5277a3297691bc1d46ea5412586f`，核对文件：

| 路径（相对仓库根目录） | SHA-256 |
| --- | --- |
| .venv/smplerx-assets/SMPLer-X/main/SMPLer_X.py | ad3215d700e917a9a6a818500bca853d1bd4d647778f24e78625a9554b5ebcf8 |
| .venv/smplerx-assets/SMPLer-X/main/inference.py | 5b6e7add52e0699c1fa1dcbec12fa67a9d7449d9acdde86a94ee814f69d97db7 |

实测本地输出：`outputs/smplerx_probe_camera_dshow_20261031/`，含现场画面，按现有规则不提交 Git。

## 实时仿真没有完成标定时

用户后续 30 帧探针：4 帧体型初始化，9 帧有效，第 14 帧开始出现两个合格人体框，
后 17 帧被身份保护拒绝。探针 `passed` 仅表示曾有有效帧，不表示整段持续成功。
框数是检测器判断，不一定代表现场真有两个人；需查看对应 `overlay.png`，
排查第二个人、屏幕/镜面中的人像或检测误报。
已查看本机 `00013/overlay.png`：画面可见一位操作者。完整人体框分数 0.993375，
另一个覆盖左侧手臂及部分身体的框分数 0.775333，二者都通过当前过滤。
这支持重复人体检测导致误触发的判断，不能将 accepted_count=2 等同于现场确有两个人。
本次没有删除重叠框或放宽多人保护；此类检测策略变化需要另外验证真实多人场景。

用户的实时报告 `outputs/tmp_realtime/arm_restart.json`：655 帧、约 64 秒，
双侧可靠新身体样本均为 0，机械臂标定均为 0/30。
手指 applied 477/434、腕位置 IK 残差和肩部角度变化不能证明人体机械臂映射有效；
`arm_targets_valid_frames=0` 才说明该次没有有效身体目标。
该旧报告没有保存工作进程原始拒绝原因，不能据此确定也是多人锁定。

后续核查发现：跟踪器对“从未接受有效样本”的无限年龄错误标记为已过期，
同时覆盖了未检测到人体、体型初始化等真实原因。已修正为只有存在已接受样本时
才执行该样本的过期状态转换；有效性判定、过期时限和身份保护保持原设置。
无效结果保留模型原始 reason。现在实时报告额外包含 `body_worker`：

- `results_received`：实际收到的结果数，不计等待轮询。
- `valid_results`：模型结果中各侧有效次数，与跟踪器可靠新样本数分开计数。
- `result_reasons`：原始结果原因及次数；空原因记为 `valid_body`。
- `fault`、`stderr_tail`：工作进程/传输异常及近期诊断输出。
- `last_result`：最近收到的序号、时间、有效标记、原因与包含传输等待的耗时。
- `tracker_reasons`：最终跟踪状态的原因。

退出时控制台也打印 `body worker: results=... valid=... reasons=... fault=...`。
再次运行用户原有实时命令即可获得这些字段，建议换一个报告文件名保留旧证据。
若需要核对现场画面，可使用已有 `--dump-body outputs/tmp_realtime/body_diag.h5` 保存采集与结果。
实时机械臂成功应同时确认 `body_source.sides.*.reliable_new_samples` 增长、标定完成、
`arm_targets_valid_frames` 大于 0，随后再评价运动质量。

相同 dex 环境下，对现场探针第一帧执行现有真实 worker roundtrip（6 帧）已通过，
H5 往返误差为 0。另一次 35 秒当前摄像头复现收到首帧输入过期及后续
`no person detected`，无工作进程故障，但旧跟踪器把它们统一显示为过期。
这不是用户先前现场状态的复现证据，不能替代该次缺失的原始原因。

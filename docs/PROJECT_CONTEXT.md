# 项目交接文档 / AI 对话上下文

> **用途**：新开一个 AI 对话时，先让它读本文件，它就能立刻了解项目全貌、环境陷阱和历史决策。
> **文件位置**：`F:\simulation_platform\docs\PROJECT_CONTEXT.md`
> **最后更新**：2026-09-12

---

## 0. 给 AI 的第一段话（复制这段给新对话）

```
请先完整阅读 F:\simulation_platform\docs\PROJECT_CONTEXT.md，
这是本项目的交接文档，包含项目背景、技术栈、运行环境陷阱、与论文的差距清单。
读完后再开始工作。
```

---

## 1. 项目概况

| 项目 | 内容 |
|---|---|
| **中文名** | 大创 —— 遥操作系统 |
| **性质** | 中山大学 · 大学生创新创业训练计划项目 |
| **核心目标** | **用 PyBullet 复现 TeleOpBench 论文**（双臂灵巧手遥操作仿真基准平台）|
| **本地路径** | `F:\simulation_platform` |
| **GitHub 仓库** | `https://github.com/Wangxianyu835/teleoperation-system` |
| **GitHub 用户名** | `Wangxianyu835` |
| **提交邮箱** | `328205274+wangxianyu835@users.noreply.github.com`（GitHub 隐私邮箱）|
| **作者** | 王宪雨（中山大学）|

### 项目规模

- **源码 2,392 行 / 23 个 Python 文件**（整个仓库仅 ~115 KB）
- **30 个分层任务**（Level 1~4）
- **3 种人形机器人模型**

---

## 2. 复现目标论文

| 项目 | 内容 |
|---|---|
| **标题** | TeleOpBench: A Simulator-Centric Benchmark for Dual-Arm Dexterous Teleoperation |
| **arXiv** | 2505.12748v2 [cs.RO]，2025-09-15 |
| **机构** | 上海AI实验室、浙大、港中文、港科大(广州)、港大、Feeling AI、清华深研院 |
| **作者** | Hangyu Li\*, Qin Zhao\*（共同一作）；Junting Dong†, Jiangmiao Pang（通讯）|
| **项目主页** | https://gorgeous2002.github.io/TeleOpBench/ |
| **本地 PDF** | `F:\26年大创 双臂遥操作平台\2025TeleOpBench.pdf`（13 页）|
| **官方参考源码** | `F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\TeleOpBench-main`（1.4 GB）|
| **许可证** | Apache-2.0（Unitree Robotics / HangZhou YuShu TECHNOLOGY CO.,LTD.）|

### 论文核心内容（速览）

**一句话**：为「双臂灵巧遥操作」建立第一个公平、可复现的基准测试平台 —— 用**仿真器固定住机器人和任务环境**，消除硬件差异，让四种遥操作接口能在完全相同条件下被对比。

**三大贡献**：
1. **30 个高保真任务环境**（覆盖运动学与力交互难度广谱）
2. **4 种代表性遥操作管线**，统一模块化接口（MoCap / VR / 外骨骼 / 单目视觉）
3. **仿真 + 真机对照实验**，10 个任务上强相关 → 证明仿真能预测真机表现

**论文的四种遥操作接口**：

| # | 接口 | 关键硬件/算法 |
|---|---|---|
| ① | Vision-based | SMPL 体型估计 + SMPLer-X 姿态 + **MediaPipe + Dex-Retargeting**（手部）+ **PINK** IK + 卡尔曼滤波 |
| ② | MoCap-based | **Xsens MVN**（17 IMU）+ **Xsens Metagloves by Manus**（每手 20 DoF）+ 关节级重缩放 + **CLIK 闭链IK** |
| ③ | VR-based | **Apple Vision Pro**（OpenXR）+ **PINK** IK + AnyTeleop dex-retargeting |
| ④ | Exoskeleton-based | **HOMIE** 同构外骨骼 + 伺服关节 + 霍尔传感器手套（每手 15 DoF）+ **直接映射绕过 IK** |

**论文的 3 种机器人**：

| 机器人 | 厂商 | 手臂 | 灵巧手 | 手 DoF |
|---|---|---|---|---|
| H1-2 | Unitree | 7 DoF/臂 | Inspire 灵巧手 | 5指 6 DoF |
| GR1-T2 | Fourier | 7 DoF/臂 | Fourier 原生灵巧手 | 5指 6 DoF |
| G1 | Unitree | — | 轻量三指手 | 3指 4 DoF |

**论文的评估协议**：从 30 个任务中选 **10 个代表性任务**，指标 = **成功率(%) + 完成时间(s)**，**4 名被试**。

**论文实验结论（各方法性格）**：

| 方法 | 优势 | 短板 |
|---|---|---|
| Vision | 部署最简单、成本最低 | 帧率低、腕部朝向估计粗糙、遮挡 → 只能做简单任务；Task 4/7 差 |
| VR | 抓取强 | **Task 7（ball bimanual 双手传递）完全失败**（手挡手致位姿估计失效）|
| Exoskeleton | 控制平滑，多数任务好 | 侧向肘部运动受限 → Task 1 耗时增加 |
| Xsens | 最平滑最精确、耗时最少 | **最贵** |

**论文局限性**：只覆盖上半身+桌面场景；**无触觉反馈**；未来方向 = loco-manipulation + haptics。

---

## 3. 项目现状（代码结构 + 已验证功能）

### 3.1 目录结构

```
F:\simulation_platform\
├── main.py                  137行  主入口（交互选任务/指定任务/demo/benchmark/无头）
├── demo_teleop.py           119行  无控制器演示
├── show_all.py              137行  ⭐ 三机器人并排展示
├── show_hand.py              71行  ⭐ LinkerHand 灵巧手展示
├── test_camera.py            22行  摄像头测试
├── test_import.py            50行  模块导入自检
├── requirements.txt                 pybullet/numpy/h5py/scipy
│
├── envs\                           仿真环境
│   ├── simulation_env.py    176行  环境核心类（集成机器人/任务/记录/随机化）
│   ├── robot_loader.py       46行  机器人模型加载
│   ├── sensor_recorder.py   123行  观测数据记录（状态/相机流/物体元数据）
│   └── domain_randomizer.py  54行  域随机化 ★论文没有，本项目额外加的
│
├── tasks\
│   ├── all_tasks.py         682行  ⭐ 30 个分层任务实现
│   ├── base_task.py         173行  任务基类
│   └── task_registry.py      32行  任务注册表
│
├── teleop\
│   ├── teleop_pipeline.py   244行  完整遥操作流水线
│   ├── vr_interface.py      140行  Vision Pro 手腕追踪 → 关节角度
│   ├── hand_interface.py     42行  灵巧手驱动（l7 的 17 个 UDP 关节）
│   ├── camera_interface.py   50行  摄像头采集（联想电脑 RGB）
│   └── pipeline_data.py     143行  流水线数据结构
│
├── utils\metrics.py          53行  成功率 / 完成时间统计
├── docs\PROJECT_CONTEXT.md         ⭐ 本交接文档
│
├── robots\from_teleopbench\        ★机器人模型（181MB，**已入库**，普通 Git）
├── lib\                            依赖目录（335MB，未入库）
├── linkerhand_sdk\                 灵巧手 SDK（1013MB，未入库，来自 gitee）
└── .venv\                          虚拟环境（未入库）
```

**源码总计：2,392 行 / 23 个 Python 文件**

### 3.2 ✅ 已验证可运行的功能

**（1）三种机器人全部能被 PyBullet 成功加载**（2026-09-12 实测，无界面 `p.DIRECT` 模式）

| 机器人 | URDF 路径 | 结果 | 总关节 | 可动关节 |
|---|---|---|---|---|
| **H1-2** | `robots/from_teleopbench/h1_2/h1_2.urdf` | ✅ OK | **55** | 51 |
| **GR1-T2** | `robots/from_teleopbench/gr1/urdf/robot.urdf` | ✅ OK | **70** | 54 |
| **G1** | `robots/from_teleopbench/g1/g1_29dof_with_hand_lock_waist.urdf` | ✅ OK | **53** | 41 |

> 55 关节与 `envs/robot_loader.py` 注释「H1-2 (55关节人形)」完全吻合。
> 加载时的 `b3Warning: No inertial data for link` 是无害警告（URDF 部分 link 未写惯性参数，PyBullet 用默认值兜底）。

**（2）`show_all.py` 三机器人并排展示**
- H1-2（红）x=-2.2 / GR1-T2（蓝）x=0 / G1（绿）x=+2.2
- 各配彩色地面圆盘 + 头顶方块 + 3D 文字标签
- 按正则 `(shoulder_pitch|upper_arm_pitch|arm_pitch)` 动态查找左右肩关节，做正弦摆动演示
- 240 Hz 仿真 + `useFixedBase=True`

**（3）传感器数据记录**（`envs/sensor_recorder.py`）对应论文观测三类：(a) 机器人状态向量 (b) 相机流（头戴第一人称 + 固定第三人称）(c) 任务级物体位姿元数据

**（4）评估指标**（`utils/metrics.py`）对应论文：成功率 + 完成时间

### 3.3 ⚠️ PyCharm 中有一个失效的运行配置

`.idea/workspace.xml` 里注册了 4 个运行入口，其中 **`show_robot.py` 在本仓库中不存在**（可能已删除或只是计划）：

```
demo_teleop.py   ✅ 存在
show_all.py      ✅ 存在
show_hand.py     ✅ 存在
show_robot.py    ❌ 文件不存在
WORKING_DIRECTORY = $PROJECT_DIR$
```

---

## 4. ★ 运行环境（有坑，必读）

### 4.1 依赖装在哪里（★ 不是标准 venv！）

本项目用 **`pip install --target`** 把依赖装进了**项目内的 `lib\` 目录**：

```
F:\simulation_platform\lib\
├── bin\
├── pybullet.cp311-win_amd64.pyd        ← pybullet 3.2.7 的本地扩展模块
├── pybullet_data\ pybullet_envs\ pybullet_robots\ pybullet_utils\
├── numpy\   numpy-2.4.6.dist-info\   numpy.libs\
├── scipy\   scipy-1.17.1.dist-info\  scipy.libs\
└── h5py\    h5py-3.16.0.dist-info\
```

**⚠️ 后果**：`lib\` **既不在 `sys.path` 里，也不在 venv 的 site-packages 里**。
直接运行 `python main.py` 会报：
```
ModuleNotFoundError: No module named 'pybullet'
```

### 4.2 ✅ 两种实测可用的运行方式

**方式 A：设置 `PYTHONPATH`（命令行推荐）**
```powershell
cd F:\simulation_platform
$env:PYTHONPATH = 'F:\simulation_platform\lib'
python main.py
```
实测通过：
```
pybullet: F:\simulation_platform\lib\pybullet.cp311-win_amd64.pyd
numpy: 2.4.6    scipy: 1.17.1    h5py: 3.16.0
```

**方式 B：直接用 `E:\python3.11.7` 解释器（全局已装齐）**
```powershell
& 'E:\python3.11.7\python.exe' main.py
```
实测通过：`ALL OK pybullet=pybullet.cp311-win_amd64.pyd numpy=2.4.6 scipy=1.17.1 h5py=3.16.0`

> PyCharm 项目 SDK 名 = `Python 3.11 (simulation_platform) (2)`，`WORKING_DIRECTORY = $PROJECT_DIR$`

### 4.3 本机 Python 环境清单

| 版本 | 路径 | 有 pybullet？ |
|---|---|---|
| 3.11（项目 venv）| `F:\simulation_platform\.venv\Scripts\python.exe` | ❌ 只有 pip/setuptools/pypdf |
| **3.11** | **`E:\python3.11.7\python.exe`** | ✅ **齐全** |
| 3.14 | `E:\python\python.exe` | ✅ pybullet（cp314）|
| 3.12 | `C:\Users\王宪雨\AppData\Local\Programs\Python\Python312\python.exe` | ❌ |

### 4.4 requirements.txt 与实际版本

```
pybullet>=3.2.0   → 实际 3.2.7
numpy>=1.24.0     → 实际 2.4.6
h5py>=3.10.0      → 实际 3.16.0
scipy>=1.10.0     → 实际 1.17.1
```

---

## 5. Git / GitHub 配置

| 项目 | 值 |
|---|---|
| Git 安装位置 | `E:\Git`（v2.55.0.windows.3）|
| `user.name` | `Wangxianyu835` |
| `user.email` | `328205274+wangxianyu835@users.noreply.github.com` |
| 其他全局配置 | `core.quotepath=false`、`core.autocrlf=true`、`init.defaultBranch=main` |
| **SSH 私钥** | **`E:\ssh\id_ed25519`** ⚠️ **唯一一份，务必备份到 U 盘/网盘！** |
| SSH 公钥 | `E:\ssh\id_ed25519.pub` |
| `known_hosts` | `E:\ssh\known_hosts` |
| **`core.sshCommand`** | `C:/Windows/System32/OpenSSH/ssh.exe -i E:/ssh/id_ed25519 -o UserKnownHostsFile=E:/ssh/known_hosts -o IdentitiesOnly=yes` |
| 远程 `origin` | `git@github.com:Wangxianyu835/teleoperation-system.git` |
| 浏览器 HTTPS 代理 | Clash Verge（**看网页**时需要开，**推代码不需要**）|

> **⚠️ 副作用**：因为密钥不在默认的 `~/.ssh` 位置，直接敲 `ssh -T git@github.com` 会失败。
> **只有 `git` 命令能正常使用**（因为配置了 `core.sshCommand`）。日常 clone/pull/push 不受影响。

### 5.1 仓库当前状态

```
main (05eddb7)
├── 9e0a94e  初始化仓库：双臂遥操作仿真平台源码
├── 071cd31  fix: show_hand.py 去除硬编码盘符路径，改为相对脚本定位
└── 05eddb7  Merge pull request #1: 修正 show_hand.py 打印的左手型号标签
```

### 5.2 `.gitignore` 已排除

```
__pycache__/  *.pyc  .venv/  venv/  .idea/  .vscode/
lib/              ← 335MB 本地依赖
linkerhand_sdk/   ← 1013MB 第三方 SDK（来自 gitee.com/ericbrunt/linkerhand_telop_python）
robots/*          ← 排除 robots 下全部子项…（共约 274MB，代码未引用）
!robots/from_teleopbench/  ← …但【放行】from_teleopbench（181MB，已入库，见第 13 节）
data/ outputs/ logs/ *.h5 *.pkl *.pt *.ckpt ...
```

### 5.3 提交历史（供参考）

| 提交 | 内容 | 大小 |
|---|---|---|
| `9e0a94e` | 首次提交：27 个文件（源码，排除了 1.8GB 依赖与资产）| 115.4 KB |
| `071cd31` | 修复 `show_hand.py` 硬编码盘符路径 | — |
| `1cad50e` | 修复打印文字 `l10v7` → `l7`（经 PR #1 合并）| 369 字节 |

---

## 6. ★ 环境陷阱（换机器 / 重装前必读）

这些都是**已经踩过并解决**的真实问题，记录下来避免重复踩坑。

### 6.1 中文用户名导致 Git 自带 ssh 失效

**现象**：
```
Could not create directory '/c/Users/\315\365\317\334\323\352/.ssh' (No such file or directory).
Failed to add the host to the list of known hosts (.../known_hosts).
git@github.com: Permission denied (publickey).
```
> `\315\365\317\334\323\352` 是「**王宪雨**」的 GBK 八进制编码（`CD F5`=王，`CF DC`=宪，`D3 EA`=雨）。

**原因**：Git 自带的 `E:\Git\usr\bin\ssh.exe` 是 **MSYS 版**，把中文用户名按 GBK 字节串去查目录，找不到真实的 `C:\Users\王宪雨\.ssh`。

**对照测试证据**：
| 使用的 ssh | 结果 |
|---|---|
| 默认（Git 自带 MSYS，`OpenSSH_10.3p1`）| ❌ `Host key verification failed` |
| 强制 Windows 自带（`OpenSSH_for_Windows_9.5p2`）| ✅ 成功 |

**解决**：设置 `core.sshCommand` 强制用 Windows 原生 ssh（见第 5 节）。

### 6.2 E 盘目录权限「不继承」的坑

**现象**：在 `E:\` 下新建的文件夹，**里面写不进文件**（访问被拒绝）。

**原因**：`E:\` 根目录给用户的 `(M)` 权限**没有 `(OI)(CI)` 继承标记**，所以新建子目录不继承写权限。

**解决**（你是目录所有者，**不需要管理员权限**）：
```powershell
icacls "E:\目标目录" /grant "*<用户SID>:(OI)(CI)M"
```
> 本机已修复：`E:\ssh`、`E:\projects`
> **一劳永逸方案**（需**管理员**运行，因为 `E:\` 根目录所有者是 `NT AUTHORITY\SYSTEM`）：
> ```powershell
> icacls "E:\" /grant "*<用户SID>:(OI)(CI)M"
> ```

### 6.3 SSH 私钥必须收紧 ACL

**原因**：Windows OpenSSH 有安全检查 —— 私钥若对其他用户可读，会直接拒绝使用（报 `UNPROTECTED PRIVATE KEY FILE`）。`E:\` 下默认权限较宽松，必须收紧。

**解决**：
```powershell
icacls "E:\ssh\id_ed25519" /inheritance:r /grant:r "*<用户SID>:F"
```

### 6.4 ★ GitHub 网页被 SNI 层阻断，但 SSH 通

**现象**：浏览器打不开 `github.com`（提示「响应时间太长」），但 `git push` 完全正常。

**实测证据**：
| 测试 | 结果 |
|---|---|
| TCP 连 GitHub 各 IP 的 **443** 端口 | ✅ 全部 OPEN |
| HTTPS 访问 `github.com`（**换任意 IP 都一样**）| ❌ 全部超时（TLS 握手被阻断）|
| HTTPS 访问 `api.github.com` | ✅ 200 |
| **SSH（22 端口）访问 `github.com`** | ✅ **OPEN** |

**结论**：这是针对 `github.com` 域名的 **SNI 定向干扰**，**不是 IP 问题** → **改 hosts / 换 IP 都无效**。

**应对策略**：
| 目的 | 方案 |
|---|---|
| **推/拉代码** | 走 **SSH**，**无需代理**，稳定可用 ✅ |
| **看 GitHub 网页** | 开 **Clash Verge** 代理（加密隧道绕过 SNI 检测）|

> ⚠️ **代理节点选择**：**不要用香港机房节点**（DataDome 会拦截），要选节点名带「**住宅 / 家宽 / 原生IP / Residential**」的。

### 6.5 DataDome 风控（注册/登录时）

GitHub 用 **DataDome** 保护 `/signup` 等接口。响应头特征：`x-datadome: protected`、`Set-Cookie: datadome=...; Max-Age=31536000`（**有效期 1 年**）。

**触发特征**：机房/代理 IP（`hosting:true`）、**脏的 `datadome` cookie**、广告拦截扩展（拦掉 `captcha-delivery.com` 验证脚本）。

**应对**：
1. **清除 `github.com` 站点数据**（F12 → Application → Storage → Clear site data）← 关键！
2. 关闭 uBlock/AdGuard 等广告拦截扩展
3. 换住宅类节点（不要机房 IP）
4. 从 `github.com` 首页自然点击进入，不要直接输 `/signup`
5. 失败后**不要连续重试**，等 24 小时

### 6.6 其他已知现象

- **pybullet 加载时的 `b3Warning: No inertial data for link`** → 无害警告，URDF 部分 link 未写惯性参数，PyBullet 用默认值兜底
- **`git log --oneline` 易打成 `--online`** → 会报 `fatal: unrecognized argument`，用 Tab 补全可避免
- **PowerShell 敲中文提交信息不会乱码**（已实测 UTF-8 正常），但**引号必须用英文 `"`**

---

## 7. ★ 与论文的差距清单（写报告/答辩直接用）

### 7.1 30 个任务 —— ✅ **完全对应**

| Level | 分类 | 数量 | 任务名（`tasks/all_tasks.py` 注册名）|
|---|---|---|---|
| **1** | 基础拾取放置 | 5 | `pushcube` `pickcube` `pickplacecube` `uprearcup` `balltrashcan` |
| **2** | 工具操作 | 11 | `rotatefaucet` `rotatehearth` `openmicrowave` `closemicrowave` `opendrawer` `closedrawer` `liftmug` `openlaptop` `ballmug` `pourwater` `breadtoaster` |
| **3** | 双手协作 | 9 | `ballbimanual` `potbimanual` `pottomato` `pottray` `stackboxes` `panhearth` `tidyuptable` `pottomatoout` `plateoven` |
| **4** | 长时域序列 | 5 | `pottomatoplate` `penbrushpot` `drawerbook` `twistbottlecaps` `stacktoyblocks` |

**与论文 Table 1 的 30 个任务 100% 对应**，唯一差异：
> 论文写作 `twist bottle cap`（**单数**），代码注册名是 `twistbottlecaps`（**复数，多了个 s**）

### 7.2 仿真器 —— ⚠️ **根本性差异（最重要的偏离）**

| 关键项 | 论文 | 本项目 |
|---|---|---|
| **仿真器** | **NVIDIA Isaac Sim** | **PyBullet** |
| 物理引擎 | PhysX | Bullet |
| 渲染 | 照片级（photorealistic）| 基础 OpenGL |
| 模型格式 | USD | URDF |

> 🚨 论文标题就是 **「Simulator-Centric」** —— **仿真器本身是论文的核心贡献**。论文明确说明选 Isaac Sim 是因为「高性能 PhysX 引擎 + 照片级渲染器」，才能让仿真结果预测真机表现。

**★ 已确认的项目决策（2026-09-12，用户明确表示）**：
> **「我就是要用 PyBullet 做」**
>
> **理由**：PyBullet 轻量、无需 GPU、无需 Isaac Sim 授权/许可，且已成功加载 3 种机器人。
>
> **⚠️ 写报告/答辩时必须声明**：本项目是 **PyBullet 简化复现**，聚焦「任务体系 + 遥操作流程 + 评测协议」，**不复现论文的物理/视觉保真度结论**（不强声称「仿真能预测真机」）。

> 💡 项目里同时存在 USD 资产 `robots/h1_with_hand/h1_with_hand_rt.usd`（59.5 MB），未来若升级到 Isaac Sim 可用。

### 7.3 四种遥操作接口 —— 覆盖 **2 / 4**

| # | 论文接口 | 本项目对应文件 | 状态 |
|---|---|---|---|
| ① | **Vision-based**<br>SMPL + SMPLer-X + MediaPipe + Dex-Retargeting + PINK + 卡尔曼 | `teleop/camera_interface.py`<br>（注释：联想电脑 RGB 摄像头采集）| ⚠️ **实现不同**（只是摄像头采集，非视觉遥操作）|
| ② | **MoCap-based**<br>Xsens MVN（17 IMU）+ Manus 手套（20 DoF）| ❌ 无 | ❌ **缺失**（需硬件）|
| ③ | **VR-based**<br>Apple Vision Pro + PINK + AnyTeleop | `teleop/vr_interface.py`<br>（Vision Pro 手腕追踪 → 关节角度）| ⚠️ **部分实现** |
| ④ | **Exoskeleton-based**<br>HOMIE 同构外骨骼 + 霍尔手套（15 DoF）| ❌ 无 | ❌ **缺失**（需硬件）|

> 论文整个实验设计是「**同一套任务 × 4 种接口**」。目前只有 1~2 种可用，**对照实验暂时无法成立**。

### 7.4 关键技术组件 —— 缺失清单

| 组件 | 用途 | 状态 |
|---|---|---|
| **dex-retargeting** | 向量优化手部重定向（论文 Vision + VR **都靠它**）| ❌ 未使用（`pip install dex-retargeting` 可装）|
| **PINK** | 逆运动学求解 | ❌ 未使用（`pip install pink`）|
| **SMPL / SMPLer-X** | 人体姿态与体型估计 | ❌ 未使用 |
| **MediaPipe** | 手部关键点检测 | ❌ 未使用（`pip install mediapipe`）|
| **卡尔曼滤波** | 关节平滑（减少抖动）| ❌ 未使用 |
| **CLIK** | 闭链逆运动学（MoCap 用）| ❌ 未使用 |
| **灵巧手 DoF 匹配** | 论文：H1-2=6DoF / GR1-T2=6DoF / G1=4DoF | ⚠️ 待核实 |

### 7.5 本项目「额外」有的（论文没有）

| 组件 | 说明 |
|---|---|
| `envs/domain_randomizer.py` | **域随机化**（光照/摩擦/纹理/位置）—— 论文明确**没有**做 sim2real 随机化，这是本项目的扩展 |
| `README.md` / `NOTICE` / `.gitignore` / `docs/` | 工程化配套文档 |
| Git / GitHub 版本管理 | 论文是发布源码包，本项目用 Git 管理全过程 |
| `linkerhand_sdk` 集成 | 引入 LinkerHand 灵巧手（论文用 Inspire / Fourier / Unitree 手）|

---

## 8. 历史会话要点（2026-09-12 首次会话）

### 8.1 时间线（按顺序）

| # | 内容 |
|---|---|
| 1 | **GitHub 注册被 DataDome 拦截** → 诊断根因（香港机房 IP `89.185.26.180` + 有效期 1 年的脏 `datadome` cookie）→ 清站点数据后注册成功 |
| 2 | 确认 Git 已安装：`E:\Git`，v2.55.0.windows.3 |
| 3 | 配置 Git 全局身份（使用 GitHub 隐私邮箱，不暴露学校邮箱）|
| 4 | 生成 **ed25519** SSH 密钥并添加到 GitHub；核对主机密钥指纹与官方一致（无中间人）|
| 5 | 分析项目体积：**1.8 GB / 12,462 文件**，发现 3 大体量目录 + 一个 **119 MB 的 pack 文件**（超 GitHub 100MB 硬限）|
| 6 | 编写 `.gitignore`（排除 lib / robots / linkerhand_sdk / .venv / .idea / __pycache__）+ `README.md` + `NOTICE`（第三方开源声明）|
| 7 | `git init` → 首次提交（**27 文件 / 115.4 KB**）→ push 成功 |
| 8 | 修复 `show_hand.py` 硬编码盘符路径（改为基于 `__file__` 动态计算）|
| 9 | **完整走完 PR 流程**：`checkout -b` → 改代码 → `diff` → `add` → `commit` → `push -u` → 网页 Create PR → Merge → `checkout main` + `pull` + `branch -d` |
| 10 | 排查并解决 4 个环境问题：**中文用户名 SSH 失效 / E 盘权限不继承 / SNI 阻断 / 私钥 ACL** |
| 11 | 把 SSH 密钥从 `C:\Users\王宪雨\.ssh` **迁移到 `E:\ssh`**（用户不想在 C 盘存东西）|
| 12 | 完整精读论文（13 页全文 + 68 篇参考文献）|
| 13 | **实测验证 3 种机器人能被 PyBullet 成功加载**（H1-2=55关节 / GR1-T2=70关节 / G1=53关节）|
| 14 | 生成本交接文档（`docs/PROJECT_CONTEXT.md`）|
| 15 | 排查「仿真环境到底能不能跑」，**发现 8 个问题**（详见第 12 节）|
| 16 | **修复接口不匹配**：重写 `RobotLoader`（3 种机器人 + 关节映射表）、修 `apply_action` 动作错位、修 `SimulationEnv`、修 `main.py` |
| 17 | **首次跑通端到端**：`test_import.py` 6 项全绿（`EXIT=0`），H1-2 + pushcube 跑 300 步无异常 |
| 18 | 新增 `docs/INTERFACE_CONTRACT.md`（接口契约）+ `docs/TEAM_ONBOARDING.md`（队友上手指南）|
| 19 | 用户告知：**重定向算法由队友（肖奕阳）负责，数据采集由另一位队友负责** |

### 8.2 已掌握的 Git 工作流

```powershell
cd F:\simulation_platform              # 1. 进项目
git pull                               # 2. 同步最新
git checkout -b feature/新功能名        # 3. 开分支
#   ... 改代码 ...
git status                             # 4. 看改了啥
git diff                               # 5. 确认改对了
git add .                              # 6. 暂存
git commit -m "feat: 做了xxx"           # 7. 提交
git push -u origin feature/新功能名     # 8. 推送
#   ... 网页开 PR → Merge ...
git checkout main                      # 9. 切回主线
git pull                               # 10. 同步合并结果 ⭐ 最容易漏
git branch -d feature/新功能名          # 11. 删本地分支
```

**提交信息规范**：`feat:` / `fix:` / `docs:` / `style:` / `refactor:` / `test:` / `chore:` + **一个空格** + 说明

---

## 9. 待办清单

### 🔴 高优先级

- [ ] **⚠️ 备份 SSH 私钥** `E:\ssh\id_ed25519` 到 **U 盘 / 网盘（加密压缩）** ← 最紧急，目前全世界只有这一份
- [ ] 确认 GitHub 仓库可见性是 **Private** 还是 Public（Settings → 最下方 Danger Zone）
- [ ] 跑通 `main.py` 的 30 个任务（目前只实测了「机器人加载」和「展示脚本」）
- [ ] 核实三种机器人挂载的灵巧手 DoF 是否与论文一致（H1-2=6 / GR1-T2=6 / G1=4）

### 🟠 中优先级

- [ ] 深化 `teleop/vr_interface.py`，对齐论文的 **PINK IK + AnyTeleop dex-retargeting**
- [ ] 引入 `dex-retargeting` 做手部重定向（论文 Vision + VR 都靠它）
- [ ] 补齐 `show_robot.py`（PyCharm 里有运行配置，但文件不存在）
- [ ] 修正 `twistbottlecaps` → 与论文一致的命名
- [ ] 处理 `teleop/camera_interface.py`：目前只是摄像头采集，还不是论文的视觉遥操作

### 🟢 低优先级 / 长期

- [ ] 用 **Git LFS** 把 `robots/from_teleopbench/`（181 MB）纳入版本管理
- [ ] 用 **GitHub Issues** 管理任务（答辩时可展示工程管理能力）
- [ ] 实现论文的评测协议：**10 个代表性任务 × 4 种接口 × 4 名被试**
- [ ] 考虑迁到 **Isaac Sim**（项目里已有 USD 资产），以对齐论文的仿真保真度

---

## 10. 常用命令速查

### 运行项目

```powershell
cd F:\simulation_platform
$env:PYTHONPATH = 'F:\simulation_platform\lib'   # ★ 必须！否则找不到 pybullet

python main.py                        # 交互式选任务
python main.py --task pushcube        # 运行指定任务
python main.py --demo                 # 演示模式
python main.py --benchmark            # Level 1 任务基准测试
python main.py --task pushcube --no-render   # 无头模式

python show_all.py                    # 三机器人并排展示
python show_hand.py                   # LinkerHand 灵巧手展示
python test_import.py                 # 模块导入自检
```

### Git

```powershell
git status                            # 看状态（最常用）
git log --oneline --graph --all --decorate   # 分支图
git diff                              # 看未暂存改动
git branch -a                         # 所有分支
git remote -v                         # 远程地址
```

### 排障

```powershell
# 报 Permission denied (publickey)
git config --global --get core.sshCommand     # 应显示 Windows OpenSSH 路径
git ls-remote origin                          # 测试 SSH 通道是否通

# GitHub 网页打不开（不是代码问题）
#   → 开 Clash Verge 代理看网页；推代码不需要代理

# 报 ModuleNotFoundError: No module named 'pybullet'
$env:PYTHONPATH = 'F:\simulation_platform\lib'   # 设置后重试
```

---

## 附：相关文件索引

| 文件 | 说明 |
|---|---|
| `F:\simulation_platform\docs\PROJECT_CONTEXT.md` | **本文件**（AI 交接文档）|
| `F:\simulation_platform\README.md` | 项目说明（会展示在 GitHub 首页）|
| `F:\simulation_platform\NOTICE` | 第三方开源声明（TeleOpBench Apache-2.0 / LinkerHand SDK）|
| `F:\simulation_platform\.gitignore` | 已排除 1.8GB 依赖与资产 |
| `F:\26年大创 双臂遥操作平台\2025TeleOpBench.pdf` | 论文原文（13 页）|
| `F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\` | 论文官方参考源码（1.4 GB）|
| `E:\ssh\id_ed25519` | ⚠️ **SSH 私钥（唯一一份，务必备份）** |

---

---

## 11. ★ 文档维护约定（后续 AI 请务必遵守）

> 本文档是**活文档**，需要随项目进展持续更新。**每次对话结束前，请检查是否需要更新以下内容。**

### 11.1 更新清单

| 章节 | 何时更新 |
|---|---|
| 文首 **「最后更新」日期** | ⚠️ **每次修改本文件都必须改** |
| **3.2 已验证可运行的功能** | 新验证成功的功能（**附实测数据**）|
| **5.1 仓库当前状态 / 5.3 提交历史** | 有新提交时 |
| **6. 环境陷阱** | 踩到新坑并解决后 |
| **7. 与论文的差距清单** | 补齐了某个接口/组件后（把 ❌ 改成 ✅）|
| **8.1 时间线** | **每次对话结束时追加一行** |
| **9. 待办清单** | 完成一项就勾掉；有新任务就加上 |
| **10. 常用命令速查** | 发现新的常用命令时 |

### 11.2 更新原则

1. **只记录事实和已实测的结论**，不要写猜测或未验证的内容
2. **命令和路径必须可直接复制使用**
3. **踩坑记录写清「现象 → 原因 → 解决」三段式**
4. **绝对不要写入任何密钥、密码、Token、SID 的实际内容**
5. 保持**中文**，风格与现有章节一致
6. 表格优先（可读性最好）

### 11.3 提交约定

- 文档类改动提交信息用 `docs:` 前缀，例如：
  ```
  docs: 更新 PROJECT_CONTEXT 的待办清单与时间线
  ```
- **文档类改动** → 可直接提交到 `main`（单人项目，简单高效）
- **功能 / 修复类改动** → 走分支 + PR（保留评审记录）

```powershell
# 文档更新的标准三步
cd F:\simulation_platform
git add docs/
git commit -m "docs: 更新了 xxx"
git push
```

---

> **说明**：本文件已纳入 Git 版本管理（提交到 `main` 分支）。
> 修改后请及时 commit + push，这样在**任何设备、任何新对话**中都能获取到最新上下文。

---

## 12. 【2026-09-12】仿真环境修复记录（★ 重要）

### 12.1 修复前的真实状态

> **各模块单独都写好了，但核心的 `SimulationEnv` 从未成功运行过一次。**

此前的「环境已搭好」是**错觉** —— 验证过的两个脚本**都没有真正用到 `SimulationEnv`**：

| 验证手段 | 为什么没发现问题 |
|---|---|
| `show_all.py`（能显示三机器人）| **绕过 `SimulationEnv`**，自己直接调 pybullet |
| `test_import.py`（全绿）| 只检查「能不能 import」，`None` 也能 import 成功 |

> 类比：发动机、轮子、方向盘都分别测试过能转，但**从来没把车拼起来开过一次**。

### 12.2 发现并修复的 8 个问题

| # | 问题 | 位置 | 症状 |
|---|---|---|---|
| 1 | **相对导入越界** | `envs/simulation_env.py:11-12` `from ..tasks import` | `ImportError: attempted relative import beyond top-level package`（项目根目录无 `__init__.py`）|
| 2 | **静默吞异常** | `envs/__init__.py` 的 `try/except → SimulationEnv = None` | 致命错误被隐藏成 `None`，运行时只报 `TypeError: 'NoneType' object is not callable` |
| 3 | **变量未定义** | `main.py:116` `if choice in ('quit','exit')` | `UnboundLocalError` → **所有命令行参数模式全崩** |
| 4 | **假阳性自检** | `test_import.py` 只检查 import | 组件是 `None` 也报 ✓ |
| 5 | **函数签名不匹配** | `SimulationEnv.reset()` 调 `load_robot(self.robot_type)`，但 `load_robot()` 不收参数 | `TypeError: takes 1 positional argument but 2 were given` |
| 6 | **返回值类型不匹配** | `load_robot()` 返回 **dict**，`SimulationEnv` 当 **int** 用 | `p.getNumJoints(dict)` 崩溃 |
| **7** | **⭐ 动作映射错位（最严重）** | `BaseTask.apply_action()` 按「第 i 个动作 → 关节索引 i」映射 | URDF **前 12 个关节是腿部**！`action[0]`（本意左肩）被送到 `left_hip_yaw_joint`（左髋）→ **手臂不动、腿乱动** |
| 8 | **渲染判断错误** | `simulation_env.py` 的 `if self.client == p.GUI` | `client` 是连接 id（0），`p.GUI` 是类型常量（1）→ 永不相等 |

### 12.3 修复后的实测结果

**三种机器人的动作空间（实测 2026-09-12）：**

| robot_type | 总关节 | 可动 | 左臂 | 右臂 | 左手 | 右手 | **动作维度** | `action[0]` 对应的关节索引 |
|---|---|---|---|---|---|---|---|---|
| `h1_2` | 55 | 51 | 7 | 7 | **12** | **12** | **38** | **13** `left_shoulder_pitch_joint` |
| `gr1_t2` | 70 | 54 | 7 | 7 | **11** | **11** | **36** | **16** `left_shoulder_pitch_joint` |
| `g1` | 53 | 41 | 7 | 7 | **7** | **7** | **28** | **22** `left_shoulder_pitch_joint` |

> ✅ `action[0]` 现在指向**肩关节**（索引 13 / 16 / 22），不再是索引 0（腿）
> ✅ 自动检查「动作是否误触腿部关节」→ **否**

**端到端测试（首次成功）：**
```
$ python test_import.py
测试6: ✓ 端到端跑通（robot_id=1, action_dim=38）
     任务对象 = ['table', 'cube', 'target']
所有测试完成！
EXIT=0                                   ← 退出码 0（真·全绿）
```

### 12.4 本次新增的文件

| 文件 | 用途 |
|---|---|
| `docs/INTERFACE_CONTRACT.md` | ⭐ **接口契约**（动作空间 / 观测空间 / 控制器接口 / HDF5 格式）—— **队友对接必读** |
| `docs/TEAM_ONBOARDING.md` | 队友上手指南（装环境 → 拿模型 → 验证，含 7 个 FAQ）|

### 12.5 新增的关键 API

```python
# RobotLoader（envs/robot_loader.py）
loader.load_robot(robot_type)          # 'h1_2' | 'gr1_t2' | 'g1'，返回 dict
loader.all_joints                      # {关节名: 索引}
loader.arm_joints['left'|'right']      # 手臂关节名列表（各 7 个）
loader.hand_joints['left'|'right']     # 灵巧手关节名列表
loader.action_joint_names              # ★ 动作向量每个位置的关节名
loader.action_joint_indices            # ★ 动作向量每个位置对应的 pybullet 关节索引
loader.describe_action_space()         # 打印完整映射表

# SimulationEnv（envs/simulation_env.py）
env.action_dim                         # ★ 动作维度（不是固定 28！）
env.action_joint_names / action_joint_indices
env.robot_id                           # int（已修正类型）

# BaseTask（tasks/base_task.py）
task.apply_action(action, joint_indices=env.action_joint_indices)   # ★ 必须传映射
```

### 12.6 ⚠️ 仍然存在的问题（待办）

| 问题 | 来源 | 影响 |
|---|---|---|
| `teleop_pipeline.py` 用 **TRON2(逐际动力) / xArm7** 的硬编码索引 | 早期代码 | 与论文 3 种机器人不符，需重写或废弃 |
| `teleop_pipeline.py` **未接入 `SimulationEnv`** | 架构缺口 | 重定向算法没有标准入口（契约 C 已定义接口，待实现）|
| `main.py` 非交互路径**仍用 `demo_controller`** | 功能缺口 | `--task xxx` 跑的不是真实遥操作 |
| obs **缺少机器人状态向量和相机流** | 与论文的差距 | 队友 B 的 P0 任务 |
| 相机第一人称 **eye 硬编码** `[0,0,1.5]`，未跟随头部 | 与论文的差距 | 队友 B 的 P0 任务 |
| 相机流**只存首尾帧** | 与论文的差距 | 队友 B 的 P0 任务 |

---

## 13. 【2026-09-12】模型资产入库方式决策（★ 重要）

### 13.1 结论

> **`robots/from_teleopbench/`（181 MB）用【普通 Git】入库，不使用 Git LFS。**

### 13.2 为什么不用 Git LFS

排查时发现一个关键事实：**Git LFS 的文件传输强制走 HTTPS**：
```
Endpoint=https://github.com/Wangxianyu835/teleoperation-system.git/info/lfs
                              ↑ HTTPS（不是 SSH）
```

而本项目网络环境的实测情况是：

| 通道 | 实测结果 |
|---|---|
| `github.com` 的 **HTTPS（443）** | ❌ **SNI 定向干扰**，直连超时；**必须开代理**才能通 |
| `github.com` 的 **SSH（22）** | ✅ **稳定可用，无需代理** |

**若用 LFS，代价是：**
- 你**和两位队友**都必须**常开 Clash 代理**才能 push/pull 模型
- 受 LFS 免费额度限制：**1 GB 存储 + 1 GB/月流量**
  （3 人各 clone 一次 ≈ 543 MB，**很容易超**，超额要付费买流量包）
- 队友还要额外安装 `git-lfs`、配置代理

**用普通 Git 的收益：**
- 走 SSH → **完全不需要代理** ✅
- 队友 **`git clone` 一步到位**，不需要 git-lfs ✅
- **无配额限制** ✅
- 代价：仓库体积 181 MB（远低于 GitHub **1 GB 警告线 / 5 GB 硬限**）

### 13.3 模型体积构成

| 子目录 | 体积 | 说明 |
|---|---|---|
| **`gr1`** | **117.4 MB** | GR1-T2（含 32 MB 的 `gr1.usd`）← **体积主体** |
| `h1_2` | 25.5 MB | H1-2 |
| `g1` | 17.4 MB | G1 |
| `unitree_hand` | 15.8 MB | |
| `inspire_hand` | 5.0 MB | |
| **合计** | **181.1 MB** | 512 个文件 |

### 13.4 `.gitignore` 的处理要点

只放行代码实际使用的 `from_teleopbench`，其余 7 个未引用的子目录（274 MB）继续排除：

```gitignore
robots/*
!robots/from_teleopbench/
```

> ⚠️ **不能用 `robots/` 整体排除** —— gitignore 规定「**父目录被排除后，无法再重新包含其子文件**」。
> 必须用 `robots/*`（只排除直接子项）+ `!robots/from_teleopbench/` 重新放行。

### 13.5 排查过程记录（供参考）

1. `git lfs version` → 已装（git-lfs 3.7.1，Git for Windows 自带）
2. `git lfs install` → 安装钩子成功
3. 创建 `.gitattributes`（按扩展名跟踪 stl/usd/glb/onnx/obj）
4. `git lfs track` → 18 条规则注册成功
5. `git check-attr filter -- ...` → 正确显示 `filter: lfs`
6. ⚠️ **测试 HTTPS**：直连 `github.com` 超时；**通过 Clash 代理返回 200**
7. → 结论：LFS 需要代理 + 有配额风险，**改用普通 Git**
8. 删除 `.gitattributes`，确认 `filter: unspecified`（LFS 规则已失效）

### 13.6 若将来必须改用 LFS

（例如模型膨胀到几十 GB）步骤：
```powershell
git lfs install
# 创建 .gitattributes 写 LFS 过滤规则
# 让 LFS 走代理：
git config --global http.https://github.com.proxy http://127.0.0.1:7897
# 然后重新 git add 目标文件（首次会被转成指针）
```

> ⚠️ **已在历史中的普通文件不会自动转成 LFS**，需要重写历史（`git filter-repo`），
> 且所有协作者都要重新 clone。**趁早决定比事后迁移便宜得多。**






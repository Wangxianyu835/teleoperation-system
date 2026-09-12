# 队友上手指南（TEAM ONBOARDING）

> 给**新加入的队友**看的操作手册。照着一步步做，约 30 分钟能跑通。
> **最后更新**：2026-09-12

---

## 第 0 步：你需要准备什么

| 项目 | 说明 |
|---|---|
| 一个 **GitHub 账号** | 没有就先注册：https://github.com/signup |
| **GitHub 用户名** | 发给王宪雨，让他把你加为协作者 |
| 电脑上装 **Git** | https://git-scm.com/download/win |
| **Python 3.11** | https://www.python.org/downloads/ |
| 磁盘空间 | 至少 **2 GB**（依赖 + 机器人模型）|

---

## 第 1 步：接受仓库邀请

1. 把你的 **GitHub 用户名**发给王宪雨
2. 他会把你加为 **Collaborator（Write 权限）**
3. 你的**邮箱**会收到邀请邮件（或登录 GitHub 后看右上角**铃铛**图标）
4. 点 **`Accept invitation`**

> ⚠️ **仓库是 Private（私有）**，所以**光有链接打不开** —— 必须先被加为协作者并接受邀请。
> 邀请**7 天过期**，过期了让王宪雨重新发。

---

## 第 2 步：配置 Git 身份

```powershell
git config --global user.name "你的名字"
git config --global user.email "你的GitHub注册邮箱"
git config --global core.quotepath false
git config --global init.defaultBranch main
```

---

## 第 3 步：配置 SSH 密钥（免密推送）

```powershell
# 1. 生成密钥（一路回车）
ssh-keygen -t ed25519 -C "你的GitHub注册邮箱"

# 2. 复制公钥内容
Get-Content ~\.ssh\id_ed25519.pub

# 3. 把上面输出的内容粘贴到：
#    https://github.com/settings/ssh/new
#    Title 随便填，Key 粘贴公钥，点 Add SSH key

# 4. 测试（应显示 Hi 你的用户名!）
ssh -T git@github.com
```

> 🔒 **私钥 `id_ed25519` 绝不能外传**；只有 `.pub` 公钥可以公开。

---

## 第 4 步：克隆仓库

```powershell
cd F:\                       # 或你想放项目的盘
git clone git@github.com:Wangxianyu835/teleoperation-system.git
cd teleoperation-system
```

> ✅ **必须用 SSH 地址**（`git@github.com:...`）。
> 用 HTTPS 地址（`https://...`）在国内网络下**经常超时推不上去**。

克隆后你会得到 **28 个文件**（只有源码，约 75 KB）。

---

## 第 5 步：搭建 Python 环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 如果报"禁止运行脚本"，先执行这一行再激活：
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

pip install -r requirements.txt
```

`requirements.txt` 内容：
```
pybullet>=3.2.0      # 仿真引擎
numpy>=1.24.0
h5py>=3.10.0         # 数据存储
scipy>=1.10.0
```

> 💡 **和项目发起人的环境不一样也没关系** —— 他用的是 `pip install --target lib`，你用**标准 venv** 即可，`python xxx.py` 直接就能跑，**不需要设置 PYTHONPATH**。

---

## 第 6 步：机器人模型 ✅ **已随仓库自动获取**

> 🎉 **好消息**：`robots/from_teleopbench/`（181 MB）**已纳入仓库**，
> 你在第 4 步 `git clone` 时**已经自动拿到了**，**不需要任何额外操作**。
>
> 验证一下：
> ```powershell
> Get-ChildItem robots\from_teleopbench -Directory
> # 应显示：g1  gr1  h1_2  inspire_hand  unitree_hand
>
> Test-Path robots\from_teleopbench\h1_2\h1_2.urdf
> # 应显示：True
> ```

**你还需要单独获取的只有这两个（体积过大/第三方）：**

| 目录 | 体积 | 何时需要 | 获取方式 |
|---|---|---|---|
| `lib/` | 335 MB | 一般不需要（用你自己的 venv 即可）| `pip install -r requirements.txt` |
| `linkerhand_sdk/` | 1013 MB | **只有跑 `show_hand.py` 时需要** | 见下方 |

```powershell
# 只有需要跑灵巧手展示脚本时才执行（1GB，可跳过）
git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git linkerhand_sdk
```

> 💡 **说明**：本项目**没有使用 Git LFS**（LFS 强制走 HTTPS，而国内网络对
> `github.com` 的 HTTPS 有干扰，需要常开代理）。改为**普通 Git + SSH**，
> 所以 clone 一次就拿到全部模型，**你也不需要安装 git-lfs、不需要配代理**。

---

## 第 7 步：验证环境 ✅

```powershell
python test_import.py
```

**应该看到（全绿）：**
```
测试1: 导入基础模块...            ✓ utils.MetricsTracker
测试2: 导入任务模块...            ✓ 已注册 30 个任务
测试3: 导入环境模块...            ✓ envs.RobotLoader / SensorRecorder /
                                    DomainRandomizer / SimulationEnv
测试4: 创建PyBullet实例...        ✓ PyBullet 连接正常
测试5: 创建仿真环境实例...        ✓ 获取任务类: pushcube -> PushCube
测试6: ★ 端到端实测...            ✓ 端到端跑通（robot_id=1, action_dim=38）
所有测试完成！
```

**再跑可视化（会弹窗看到三台机器人）：**
```powershell
python show_all.py
```

> 看到 H1-2（红）/ GR1-T2（蓝）/ G1（绿）并排摆动手臂 = **环境搭建成功** 🎉

---

## 第 8 步：开始干活（先读契约）

**⚠️ 动代码之前，必读：**
```
docs/INTERFACE_CONTRACT.md     ← 接口契约（动作空间/观测空间/控制器接口/数据格式）
docs/PROJECT_CONTEXT.md        ← 项目全貌（含环境陷阱、与论文差距）
```

### 分支约定

```powershell
# 每天开工先同步
git checkout main
git pull

# 从 main 切出你自己的分支
git checkout -b alg/retarget-xxx      # 队友A（重定向算法）
git checkout -b data/recorder-xxx     # 队友B（数据采集）
```

| 谁 | 分支前缀 |
|---|---|
| 王宪雨（仿真平台）| `dev/simulation-*` |
| 队友A（重定向算法）| `alg/retarget-*` |
| 队友B（数据采集）| `data/recorder-*` |

### 提交推送

```powershell
git add .
git commit -m "feat: 实现了xxx"
git push -u origin 你的分支名      # 第一次
git push                          # 以后
# 然后到 GitHub 网页开 Pull Request
```

---

## 第 9 步：跑「离线 Vision 流水线」（本项目的核心工作流）

> 📄 **完整规范见 [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md)**
> 📄 **接口定义见 [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) 契约 G / H**

### 三方分工

```
队友B 采集手部数据  →  队友A 做重定向  →  仿真平台回放
   (契约G)              (契约H)            (replay_actions.py)
```

### 现成可用的脚本（不需要等队友）

```powershell
# ① 查看某机器人的动作空间定义（38/36/28 维的完整关节映射）
python scripts/replay_actions.py --describe --robot h1_2

# ② 用假数据验证整条链路（不需要真实数据！）
python scripts/replay_actions.py --dummy --robot h1_2 --task pushcube --no-render

# ③ 导出一份「契约H 示例文件」，可以直接打开看格式
python scripts/replay_actions.py --dummy --save-actions datasets/samples/actions_demo_h1_2.h5

# ④ 回放真实数据（会自动读取文件里的 robot_type / task_name）
python scripts/replay_actions.py --file datasets/actions/你的数据.h5

# ⑤ 带可视化（会弹 pybullet 窗口，能看见机器人动）
python scripts/replay_actions.py --file 你的数据.h5 --render

# ⑥ 生成契约 G/H 的示例文件（照着写自己的导出代码）
python scripts/make_sample_data.py --kind all
```

### 示例数据（已在仓库里）

```
datasets/samples/
├── human_hand_demo_right.h5     契约G 示例：(T,21,3) 手部关键点
├── actions_demo_h1_2.h5         契约H 示例：38 维动作序列
├── actions_demo_gr1_t2.h5       契约H 示例：36 维
└── actions_demo_g1.h5           契约H 示例：28 维
```

> 💡 用 HDF5 查看器（如 HDFView）或 `python -c "import h5py; ..."` 打开它们，
> 就能看到自己的导出代码应该长什么样。

### 数据交换

| 数据类型 | 大小 | 方式 |
|---|---|---|
| `human_hand.h5` / `actions.h5` | < 10 MB | ✅ **git commit 到 `datasets/`** |
| 原始视频 `.mp4` | 大 | ⚠️ 网盘/移动硬盘（**不进 Git**）|

```powershell
git checkout -b data/vision-record-xxx
git add datasets/raw/
git commit -m "data: 采集 xxx 手部数据（契约G）"
git push -u origin data/vision-record-xxx
# 然后在 GitHub 开 PR 让队友A 评审
```

---

## ⚠️ 关于 Windows 编码的一个重要提醒

**中文 Windows 控制台默认 GBK，无法显示某些 Unicode 符号**（如 `✓` `✗` `⚠️`）。
本项目**已经把所有这类符号替换为 ASCII 安全写法**（`[OK]` / `[FAIL]` / `注意`）。

> 如果你在自己新增的代码里想用这类符号，**请注意**：
> - 直接 `print("✓")` 在 GBK 控制台会抛 `UnicodeEncodeError` **导致崩溃**
> - 建议用 `[OK]` 代替 `✓`，或先设置 `$env:PYTHONIOENCODING='utf-8'`
>
> 自查方法：`python -c "print('\u2713')"` —— 不报错才说明你的环境支持。

---

## 常见问题（FAQ）

### Q1：打开 GitHub 链接显示 404？

- **最常见原因**：还没接受协作者邀请，或用户名给错了
- 也可能是**没登录** GitHub 账号
- 还有一种情况：**GitHub 网页在你的网络下打不开**（见 Q2）

### Q2：GitHub 网页打不开 / 响应时间太长？

这是**国内访问 GitHub 的常见问题**（针对 `github.com` 的 SNI 定向干扰）。

| 目的 | 方案 |
|---|---|
| **git push / pull** | ✅ 走 SSH，**不需要代理**，稳定可用 |
| **看网页 / 开 PR** | 需要开代理（如 Clash），**选「住宅/家宽/原生IP」节点**，别用机房 IP |

### Q3：`git push` 报 `Permission denied (publickey)`？

```powershell
# 1. 确认公钥加到 GitHub 了
ssh -T git@github.com

# 2. 确认用的是 SSH 地址
git remote -v      # 应该是 git@github.com:...  不是 https://...
```

### Q4：Windows 用户名是**中文**，SSH 报奇怪的编码错误？

症状：
```
Could not create directory '/c/Users/\315\365\317\334\323\352/.ssh'
```
（`\315\365...` 是中文用户名的 GBK 编码）

**原因**：Git 自带的 MSYS 版 ssh 处理不了中文用户名路径。

**解决**（强制使用 Windows 原生 OpenSSH）：
```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"
```

### Q5：`ModuleNotFoundError: No module named 'pybullet'`？

- 如果你用**标准 venv**：确认已 `Activate.ps1` 且 `pip install -r requirements.txt` 成功
- 如果克隆的是王宪雨的机器环境：需要 `$env:PYTHONPATH = 'F:\simulation_platform\lib'`

### Q6：`FileNotFoundError: 找不到机器人模型`？

→ 第 6 步的 `robots/from_teleopbench/` 没放对位置，检查路径拼写。

### Q7：pybullet 刷屏 `b3Warning: No inertial data for link`？

**无害警告** —— URDF 部分 link 没写惯性参数，pybullet 用默认值兜底。可忽略。

---

## 附：项目运行入口速查

```powershell
python test_import.py                      # 环境自检（最常跑）
python main.py                             # 交互式选任务
python main.py --task pushcube             # 跑指定任务
python main.py --task pushcube --no-render # 无头模式（快）
python main.py --robot gr1_t2 --task pushcube   # 换机器人
python main.py --benchmark --trials 3      # Level 1 基准测试
python show_all.py                         # 三机器人并排展示
python show_hand.py                        # LinkerHand 灵巧手展示
```

| 参数 | 可选值 | 说明 |
|---|---|---|
| `--robot` | `h1_2` / `gr1_t2` / `g1` | 机器人类型（默认 h1_2）|
| `--task` | 30 个任务名之一 | 见 `python main.py` 的列表 |
| `--no-render` | — | 无 GUI（服务器/快速测试用）|
| `--no-record` | — | 不记录数据 |
| `--trials N` | 整数 | 基准测试每个任务次数 |

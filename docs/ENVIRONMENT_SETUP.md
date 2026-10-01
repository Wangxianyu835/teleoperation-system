# 环境配置与工具链备忘（ENVIRONMENT SETUP）

> **用途**：记录本项目的**一次性环境配置**与**踩过的坑**。
> 新开 AI 对话 / 换电脑 / 重装系统时，读这一份就能把环境复原。
>
> **配套文档**
> - 项目背景、技术栈、决策历史 → [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
> - 队友从零上手（30 分钟跑通） → [`TEAM_ONBOARDING.md`](TEAM_ONBOARDING.md)
> - 离线数据流水线 → [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md)
> - 模块接口契约 → [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md)
>
> **文件位置**：`F:\simulation_platform\docs\ENVIRONMENT_SETUP.md`
> **最后更新**：2026-10-01

---

## 0. 30 秒速查（最常忘的 6 件事）

| # | 记住这一条 |
|---|---|
| 1 | **跑项目用的是全局 Python `E:\python3.11.7\python.exe`**，<br>不是项目里的 `.venv`（那是空的，见 §2.2）|
| 2 | **`.py` 文件里禁止非 GBK 字符**（`✓ → ⚠ ²` 等），会 `UnicodeEncodeError` **崩溃**；<br>`.md` 文档不受限（见 §6）|
| 3 | **`github.com` 的 HTTPS 被阻断**：`git clone` / `push` 走 **SSH** 没问题；<br>但**看网页、点 Merge 要开代理**（见 §4）|
| 4 | **SSH 必须用 Windows OpenSSH + 显式指定 `E:/ssh` 密钥**<br>（用户名是中文，Git 自带的 MSYS ssh 会失败，见 §3.2）|
| 5 | **不要提交** `lib/`、`linkerhand_sdk/`、`.venv/`、`tmp_*/`<br>（已在 `.gitignore` 排除，见 §7）|
| 6 | **★ 任何东西都不许写进 C 盘**：依赖、pip / HF / matplotlib / torch 缓存、临时文件<br>一律落 `E:\cache\*`；**新终端先跑** `. .\scripts\env_e_drive_cache.ps1`，<br>自检 `python scripts/check_no_c_drive.py --strict`（见 §2.6）|

---

## 1. 关键路径一览

| 项目 | 路径 | 说明 |
|---|---|---|
| 项目根目录 | `F:\simulation_platform` | 本地开发目录 |
| **运行时 Python** | `E:\python3.11.7\python.exe` | ★ **项目实际使用的解释器**（3.11.7）|
| 项目内 `.venv` | `F:\simulation_platform\.venv` | ⚠️ **空的**，不能跑项目，见 §2.2 |
| **SSH 密钥目录** | `E:\ssh\` | `id_ed25519` / `id_ed25519.pub` / `known_hosts` |
| Windows OpenSSH | `C:\Windows\System32\OpenSSH\ssh.exe` | 版本 9.5p2 |
| 依赖副本 | `F:\simulation_platform\lib\` | `--target` 安装，约 335 MB，**不入库** |
| 手部 SDK | `F:\simulation_platform\linkerhand_sdk\` | 约 1013 MB，**不入库** |
| 机器人模型 | `F:\simulation_platform\robots\from_teleopbench\` | 约 181 MB，**★ 已入库** |
| 论文 PDF | `F:\26年大创 双臂遥操作平台\2025TeleOpBench.pdf` | 13 页 |
| 论文官方源码 | `F:\26年大创 双臂遥操作平台\TeleOpBench-main (1)源代码\TeleOpBench-main` | 1.4 GB |

---

## 2. Python 环境（★ 最容易踩坑的一节）

### 2.1 实际使用的解释器

**项目跑在全局 Python 上，不在虚拟环境里。**

```
E:\python3.11.7\python.exe          ← Python 3.11.7
```

它自带的依赖（已实测确认）：

| 包 | 版本 | 备注 |
|---|---|---|
| numpy | **2.4.6** | |
| h5py | **3.16.0** | |
| scipy | **1.17.1** | |
| pybullet | **3.2.7** | API 版本 202010061 |

验证命令：

```powershell
E:\python3.11.7\python.exe -c "import numpy,h5py,scipy,pybullet; print(numpy.__version__, h5py.__version__, scipy.__version__, pybullet.__file__)"
```

**所有文档里写的 `python main.py` 都默认走这个解释器** —— 请确保终端 `PATH` 里它优先于其它 Python。

### 2.2 ⚠️ 项目内的 `.venv` 是空的（不是项目环境！）

`F:\simulation_platform\.venv` 的 `site-packages` 里**只有这些**：

```
pip  setuptools  PIL(Pillow)  pypdf
```

**没有 numpy / pybullet / h5py / scipy。**

```powershell
# 会直接报错：
& .\.venv\Scripts\python.exe -c "import numpy"
#   ModuleNotFoundError: No module named 'numpy'
```

> **它是什么？** 从 `Pillow + pypdf` 这个组合推测，是当时为了
> **提取论文 PDF 文字、处理图片**而临时建的，**从未装过项目依赖**。
>
> **怎么处理？** 三种选择：
> 1. **（推荐）不用它** —— 直接把全局 Python 的路径写进 IDE 配置
> 2. 删掉它：`Remove-Item -Recurse -Force .venv`
> 3. 补齐依赖（需要能连 PyPI，可能要代理）：
>    ```powershell
>    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
>    ```
>
> ⚠️ **注意**：终端提示符显示的 `(.venv)` 只是**激活状态**，
> 不代表这个环境能用 —— 它现在是「激活着的空壳」，很有迷惑性。

### 2.3 `lib/` 目录是什么

由下面这条命令生成（`--target` 安装到指定目录）：

```powershell
pip install -r requirements.txt --target lib
```

内容 = 依赖的**一份副本**：

```
lib\
├── pybullet.cp311-win_amd64.pyd    ← 单文件扩展模块（注意：不是 pybullet\ 包目录！）
├── pybullet_data\  pybullet_envs\  pybullet_examples\
├── pybullet_robots\  pybullet_utils\
├── numpy\  numpy.libs\  numpy-2.4.6.dist-info\
├── h5py\  h5py-3.16.0.dist-info\
├── scipy\  scipy.libs\  scipy-1.17.1.dist-info\
└── bin\
```

**用法**（可选，让 `lib/` 参与模块搜索）：

```powershell
$env:PYTHONPATH = 'F:\simulation_platform\lib'
python main.py
```

> ⚠️ **两个坑**：
> 1. `lib\pybullet.cp311-win_amd64.pyd` 是**单文件** `.pyd`，所以用
>    `Test-Path lib\pybullet` 会返回 `False` —— 这不代表缺包。
> 2. 它含 Windows 专用 `.pyd`，**换 Linux/macOS 无效**。

### 2.4 四种运行方式对比

| 方式 | 命令 | 可用性 |
|---|---|---|
| **全局 Python**（推荐） | `python main.py`（PATH 指向 `E:\python3.11.7`）| ✅ 正常 |
| 全局 Python 绝对路径 | `E:\python3.11.7\python.exe main.py` | ✅ 最稳妥 |
| 全局 Python + `lib/` | `$env:PYTHONPATH='F:\simulation_platform\lib'; python main.py` | ✅ 可隔离 |
| ❌ 项目 `.venv` | `.\.venv\Scripts\python.exe main.py` | ❌ **缺依赖，不可用** |

### 2.5 环境自检（换机器 / 重装后先跑这个）

```powershell
cd F:\simulation_platform
E:\python3.11.7\python.exe test_import.py
```

应当看到 **6 项全部 `[OK]`**，最后是：

```
测试6: ★ 端到端实测（无头跑 300 步：H1-2 + pushcube）...
  加载 Unitree H1-2 (55关节, 双臂7DoF, 每手12关节)
    ★ 动作空间维度 = 38
  [OK] 端到端跑通（robot_id=1, action_dim=38）

==================================================
所有测试完成！
```

> 💡 **小提示**：pybullet 启动时会往 **stderr** 打印一行
> `pybullet build time: ...`。PowerShell 会把它标成红色错误，
> 但**这不是报错**，程序是正常的。

### 2.6 ★ 禁止往 C 盘写东西（缓存 / 临时文件全部重定向到 E 盘）

**规则（硬约束）**：项目相关的一切 —— 依赖、包管理器缓存、模型缓存、临时文件 ——
**都不许落在 C 盘**。用户明确要求过（C 盘只剩 ~18.7 GB），并且这里**真实翻车过一次**。

#### 翻车记录（2026-10-01）

| 时间 | 发生了什么 | 落点 |
|---|---|---|
| 14:40 | `pip install -r requirements-retargeting.txt` 装 torch / timm / urchin 等；安装链带出 `huggingface_hub` + `hf_xet` | 包体在 `E:\python3.11.7\...` ✅，但 `hf_xet` 日志写进了 `C:\Users\王宪雨\.cache\huggingface\xet\logs\` ❌ |
| 14:49 | 再装 `roboticstoolbox-python 1.4.4` 全家桶 | 包体在 E 盘 ✅ |
| 14:50 | **pip 的下载缓存默认就在 C 盘** | ❌ `C:\Users\王宪雨\AppData\Local\pip\Cache` 当日新增 **234 个文件 / 216.78 MB**（该目录累计 1560 MB）|

**根因**：这些工具**默认缓存目录全在 C 盘**（`%LOCALAPPDATA%\pip`、`~\.cache\huggingface`、
`~\.cache\matplotlib`…），而环境变量没被改写。更隐蔽的是 `robot_descriptions`
（RTB 取机器人 URDF 用）会往 `~\.cache\huggingface\hub` **联网下载机器人模型** —— 只要跑一次手臂 RTB 就会往 C 盘下模型。

#### 已做的修复（本机已完成）

```powershell
# 1) 建 E 盘缓存根目录（E 盘有既有权限坑，先补 ACL；见 §5）
icacls "E:\cache" /grant "*S-1-5-21-2886930988-4104668371-3581590174-1002:(OI)(CI)(M)" /T

# 2) 把已有的 C 盘 pip 缓存整体搬过去（搬，不删内容；缓存可继续复用）
robocopy "$env:USERPROFILE\AppData\Local\pip\Cache" "E:\cache\pip" /E /MOVE /R:1 /W:1

# 3) 之后一律用脚本设环境变量（可加 -Persist 写进用户级变量）
. .\scripts\env_e_drive_cache.ps1 -Persist
```

搬迁结果（实测）：`E:\cache\pip` = **889 个文件 / 1560.54 MB**，`C:\Users\王宪雨\AppData\Local\pip` 已空；
`pip cache list` 仍能看到搬迁来的 wheel（如 `pybullet-3.2.7-*.whl` 67.8 MB），`pip cache info` 显示
`Package index page cache location: e:\cache\pip\http`（1072.3 MB / 586 files）→ **缓存没有浪费，只是换了盘**。

#### 环境变量对照表

| 变量 | 指向 | 管什么 |
|---|---|---|
| `PIP_CACHE_DIR` | `E:\cache\pip` | pip 下载缓存（HTTP + wheels）|
| `MPLCONFIGDIR` | `E:\cache\matplotlib` | matplotlib 字体/配置缓存 |
| `HF_HOME` | `E:\cache\huggingface` | HuggingFace hub / xet（`robot_descriptions` 下机器人模型）|
| `TORCH_HOME` | `E:\cache\torch` | `torch.hub` 权重 |
| `XDG_CACHE_HOME` | `E:\cache\xdg` | 各类遵守 XDG 的工具 |
| `TEMP` / `TMP` | `E:\cache\tmp` | 临时文件（含 IDE / 编译中间产物）|

#### 日常用法（★ 每个新终端都要做一次）

```powershell
cd F:\simulation_platform
. .\scripts\env_e_drive_cache.ps1          # 只影响当前终端（推荐）
. .\scripts\env_e_drive_cache.ps1 -Check   # 只看当前状态，不做修改
. .\scripts\env_e_drive_cache.ps1 -Persist # 额外写入用户级变量（新终端自动生效）

# 自检：确认没有任何东西会落到 C 盘
E:\python3.11.7\python.exe scripts\check_no_c_drive.py --strict
```

`env_e_drive_cache.ps1` 会自动建目录；若建目录失败（E 盘权限不继承，见 §5），
它会**用当前用户 SID 自动补 `(OI)(CI)(M)` 再重试**，不需要手动干预。

`check_no_c_drive.py` 判定：

| 输出 | 含义 |
|---|---|
| `[OK]` | 变量指到非 C 盘，且 C 盘默认缓存目录不存在或为空 |
| `[FAIL]` | 变量未设置（工具会自己回落到 C 盘）/ 仍指向 C 盘 / C 盘上残留**非空**缓存目录 |
| `[WARN]` | C 盘残留目录为空（无害，建议删掉）|

> ⚠️ **已知残留（无害）**：IDE（VS Code + 扩展）在命令输出过大时会写
> `C:\Users\王宪雨\AppData\Local\Temp\cline\*.log`（每个几 KB）。这是**扩展自身行为**，
> 环境变量改不到它；能做的就是别让单条命令刷出巨量输出，并定期清该目录。
> 官方 Python 的 `TEMP` 已指向 E 盘，项目**编译/下载产物**不会再进 C 盘。


---

## 3. Git 与 SSH 配置

### 3.1 全局身份配置（当前状态）

```ini
[user]
    name  = Wangxianyu835
    email = 328205274+wangxianyu835@users.noreply.github.com
[core]
    quotepath = false          # 中文文件名不转义成 \344\275\240 这种八进制
    autocrlf  = true           # Windows 常规设置，但有副作用，见 §3.3
    sshCommand = C:/Windows/System32/OpenSSH/ssh.exe -i E:/ssh/id_ed25519 -o UserKnownHostsFile=E:/ssh/known_hosts -o IdentitiesOnly=yes
[init]
    defaultBranch = main
```

设置方法：

```powershell
git config --global user.name  "Wangxianyu835"
git config --global user.email "328205274+wangxianyu835@users.noreply.github.com"
git config --global core.quotepath false
git config --global init.defaultBranch main
```

### 3.2 ★ SSH 配置：中文用户名 + 网络受限的双重坑

**当前配置（`core.sshCommand`）：**

```
C:/Windows/System32/OpenSSH/ssh.exe -i E:/ssh/id_ed25519 -o UserKnownHostsFile=E:/ssh/known_hosts -o IdentitiesOnly=yes
```

| 配置项 | 为什么必须这样写 |
|---|---|
| **用 Windows 自带的 `OpenSSH\ssh.exe`** | Windows 用户名是中文 `王宪雨`。Git 自带的 **MSYS ssh 处理中文家目录会失败**，必须换用 Windows OpenSSH |
| **`-i E:/ssh/id_ed25519`** | 密钥**故意不放** `C:\Users\王宪雨\.ssh` —— 那里曾因**中文路径 + 权限问题**不可用，统一挪到 `E:\ssh` |
| **`-o UserKnownHostsFile=E:/ssh/known_hosts`** | 同上，`known_hosts` 也一并挪走，避免回到中文路径 |
| **`-o IdentitiesOnly=yes`** | 强制只用指定的这把钥匙，防止 ssh 逐个试其它身份 → 触发 GitHub 的 `Too many authentication failures` |

**设置命令：**

```powershell
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe -i E:/ssh/id_ed25519 -o UserKnownHostsFile=E:/ssh/known_hosts -o IdentitiesOnly=yes"
```

**验证连通性：**

```powershell
ssh -T git@github.com
# 期望输出：Hi Wangxianyu835! You've successfully authenticated, but GitHub does not provide shell access.
```

### 3.3 ⚠️ `core.autocrlf=true` 与 `.gitattributes` 的配合

`core.autocrlf=true` 是 Windows 常规设置，但有个**隐蔽副作用**：

> Git 靠「文件里有没有 NUL 字节」来猜是文本还是二进制。
> 而 **ASCII 格式的 `.obj` / `.stl` 网格文件不含 NUL** → 被误判成文本
> → Git 把 CRLF 换成 LF 再存 → **模型文件字节被改动**，
> 可能与 URDF 解析器不兼容，且每次 checkout 都产生大批无意义改动。

**解决办法**：仓库根的 `.gitattributes` 里显式把二进制资产标成 `binary`：

```gitattributes
robots/**/*.stl    binary
robots/**/*.obj    binary
robots/**/*.usd    binary
robots/**/*.glb    binary
...
```

> ★ **维护提醒**：以后**新增任何二进制资产类型**（如 `.fbx`、`.bin`），
> **记得同步往 `.gitattributes` 加一行 `binary`** —— 否则模型会悄悄被改坏。

### 3.4 为什么不用 Git LFS（且**千万别打开**）

当前状态：

| 项 | 状态 |
|---|---|
| `git-lfs` 是否安装 | ✅ 已装，版本 **3.7.1** |
| 全局 config 里有没有 `filter.lfs.*` | ✅ 有 4 条 |
| 仓库里有没有文件被标成 `filter=lfs` | ❌ **没有** |
| 结论 | **LFS 处于「惰性」状态，当前所有大文件都是普通 Git 对象** |

**不用的原因**：LFS 的文件传输**强制走 HTTPS**，而本网络环境对
`github.com` 的 HTTPS 存在干扰（见 §4）；SSH 才稳定。

> 🚫 **绝对不要**在 `.gitattributes` 里加 `filter=lfs`！
> 一旦加上，Git 会立刻要求通过 LFS 传输（即 HTTPS），
> **队友拉取会直接失败**。

### 3.5 ⚠️ 中文提交信息显示「乱码」是**假象，不是 bug**

**现象**：`git log` 里中文提交信息显示成

```
docs: 琛ュ叏 Apache-2.0 LICENSE 鏂囦欢锛堢涓夋柟妯″瀷璧勪骇鍐嶅垎鍙戝悎瑙勮姹傦級
```

**原因**：PowerShell 用 **GBK (cp936)** 去解码 git 输出的是 **UTF-8** 字节 → 必然乱码。
**仓库里存的是完全正确的 UTF-8。** 不用改，也不用重新提交。

**正确的验证方法**（绕开控制台编码，直接看原始字节）：

```python
import subprocess
r = subprocess.run(['git', '-C', r'F:\simulation_platform', 'log', '-3',
                    '--format=%h | %ae | %s'], capture_output=True)
print(r.stdout.decode('utf-8'))     # 必须显式按 utf-8 解码
```

期望输出（正确的中文）：

```
a1cf462 | 328205274+wangxianyu835@users.noreply.github.com | docs: 补全 Apache-2.0 LICENSE 文件（第三方模型资产再分发合规要求）
```

> 💡 **通用建议**：在中文 Windows 上排查中文乱码时，**不要相信控制台显示**，
> 一律用「捕获原始字节 + 显式指定解码方式」来判断。

---

## 4. 网络环境备忘

### 4.1 实测连通性

| 目标 | 协议 / 端口 | 状态 |
|---|---|---|
| `github.com`（网页、HTTPS）| 443 | ❌ **被阻断**（按 **SNI** 识别 `github.com`）|
| `github.com`（git over SSH）| **22** | ✅ **正常** |
| `api.github.com` | 443 | ✅ 正常 |
| `www.apache.org` | 443 | ❌ 不通（需代理）|

### 4.2 应对策略

| 你要做的事 | 怎么做 |
|---|---|
| `git clone` / `pull` / `push` | ✅ **一律用 SSH**：`git@github.com:Wangxianyu835/teleoperation-system.git` —— **不需要代理** |
| 看仓库网页 / 建 Issue / 点 Merge / 改设置 | ⚠️ **需要开代理** |
| GitHub Actions 跑 CI | ✅ 不需要（跑在 GitHub 服务器上）|
| `pip install`（PyPI）| 视情况，可能需要代理或换国内镜像 |

> 📌 **远程 URL 必须是 SSH 形式**，检查方法：
> ```powershell
> git -C F:\simulation_platform remote -v
> # 期望：origin  git@github.com:Wangxianyu835/teleoperation-system.git (fetch/push)
> ```
> 如果是 `https://github.com/...` → 改掉：
> ```powershell
> git -C F:\simulation_platform remote set-url origin git@github.com:Wangxianyu835/teleoperation-system.git
> ```

### 4.3 代理使用注意

- 代理软件：**Clash**，默认混合端口 **`7897`**
- ⚠️ **`curl.exe` 不会自动使用 Windows 系统代理** —— 必须显式指定：

```powershell
curl.exe -x http://127.0.0.1:7897 https://www.apache.org/licenses/LICENSE-2.0.txt
```

- 判断代理是否在运行：

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'clash|mihomo|verge' } | Select-Object ProcessName, Id
# 系统代理开关（1=开 / 0=关）：
(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings').ProxyEnable
```

> 💡 **踩过的坑**：下载 Apache-2.0 许可证文本时，未开代理 → `HTTP=000`；
> 最后是从**本地某个 Python 包自带的 LICENSE 副本**里取到的完整正文
> （`anytree` 包内附完整 Apache-2.0）。

---

## 5. Windows 文件权限的坑（E 盘）

### 5.1 症状

- `git clone` 到 `E:\` 或 `F:\` 时报 **`Permission denied`**
- ssh 读取 `E:\ssh\id_ed25519` 失败，或报
  `UNPROTECTED PRIVATE KEY FILE` / `Load key ...: bad permissions`
- 新建的文件夹**无法写入**，但用资源管理器「属性 → 安全」看不出明显异常

### 5.2 原因

- `E:\` 根目录上原有的 `(M)` 权限项**是显式 ACE，不带继承标记**（没有 `(OI)(CI)`）
- 于是**新建的子文件夹不会继承当前用户的写权限**
- 又因为用户名是**中文** `王宪雨`，权限项显示为 `LAPTOP-2DCCKN0R\王宪雨`，
  在某些工具里匹配会失败

### 5.3 修复命令

```powershell
# 用「用户名」授权（直观，但中文名有风险）
icacls "E:\ssh" /grant "王宪雨:(OI)(CI)(M)" /T

# ✅ 更稳妥：直接用「当前用户的 SID」授权（不受中文名影响）
icacls "E:\ssh" /grant "*S-1-5-21-2886930988-4104668371-3581590174-1002:(OI)(CI)(M)" /T
```

参数含义：

| 参数 | 含义 |
|---|---|
| `(OI)` | Object Inherit —— 子**文件**继承 |
| `(CI)` | Container Inherit —— 子**文件夹**继承 |
| `(M)` | Modify —— 允许读写改 |
| `/T` | 递归应用到所有子项 |

**查当前用户 SID：**

```powershell
whoami /user
# → laptop-2dcckn0r\王宪雨  S-1-5-21-2886930988-4104668371-3581590174-1002
```

### 5.4 当前状态（已修复）

```
E:\ssh LAPTOP-2DCCKN0R\王宪雨:(OI)(CI)(M)          ← ★ 本次修复加上的，带继承标记
       BUILTIN\Users:(I)(OI)(CI)(RX)
       NT AUTHORITY\Authenticated Users:(I)(OI)(CI)(RX)
       BUILTIN\Administrators:(I)(OI)(CI)(F)
       NT AUTHORITY\SYSTEM:(I)(OI)(CI)(F)
```

> 📌 **已修复的目录**：`E:\ssh`、`E:\projects`、`E:\cache`（2026-10-01，缓存根目录）
> 以后往 E 盘新建目录遇到权限问题时，照 §5.3 再做一次即可。

---

## 6. 编码约定：`.py` 必须是 GBK 安全的

### 6.1 规则（★ 违反会导致程序**崩溃**）

中文 Windows 控制台默认编码是 **GBK (cp936)**。如果 `.py` 里 `print()` 了
GBK 编不出来的字符，Python 会**直接抛异常并崩溃**：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\u2713'
```

> ⚠️ 注意：这是**崩溃**，不是「显示乱码」。所以必须提前避免。

### 6.2 项目的 ASCII 替代约定

| 危险字符 | 码点 | 项目约定替代写法 |
|---|---|---|
| ✓ | U+2713 | `[OK]` |
| ✗ | U+2717 | `[FAIL]` |
| ⚠ | U+26A0 | `注意` |
| → | U+2192 | `->` |
| ← | U+2190 | `<-` |
| ↔ | U+2194 | `<->` |
| ≥ / ≤ / ≈ | U+2265 / 2264 / 2248 | `>=` / `<=` / `~=` |
| ² / ³ / ¹ | U+00B2 / B3 / B9 | `^2` / `^3` / `^1` |
| ◀ / ▶ | U+25C0 / 25B6 | `<--` / `-->` |
| ️（变体选择符）| U+FE0F | 删除 |

### 6.3 `.md` 文档不受此限制

`scripts/check_gbk_safe.py` **只扫描 `.py` 文件**：

```python
for fn in files:
    if not fn.endswith('.py'):
        continue        # ← .md / .txt / .json 全部跳过
```

所以**文档里可以放心使用** `→` `✅` `⚠️` `★`（本文件就是）。
只有**代码文件**受 GBK 约束。

### 6.4 检查方法（提交前务必跑一次）

```powershell
python scripts/check_gbk_safe.py            # 只报告
python scripts/check_gbk_safe.py --strict   # 有违规则退出码 1（可用于 CI / pre-commit）
```

期望输出：

```
  未发现 GBK 无法编码的字符  [OK]
```

> 📌 **历史记录**：曾经全项目清扫过两轮 —— 第一轮 **11 个文件、66 处**；
> 后续新增代码又引入 10 类字符，再次清扫。此后该脚本正式收进仓库，
> 作为**提交前例行检查**。
>
> 📌 **最近一次修复（2026-09-30）**：`teleop/native_hand.py:173`
> 的 `map_frame()` 文档字符串里残留一个 `⚠️`（U+26A0 + U+FE0F）。
>
> 它**位于 docstring 而非 `print()`**，所以平时不崩（因为没人打印这个 docstring）；
> 但已实测：一旦执行 `help(map_frame)` 或 `print(map_frame.__doc__)`，
> 中文控制台立刻抛
> `UnicodeEncodeError: 'gbk' codec can't encode character '\u26a0'`
> —— 是**潜伏的地雷**。
>
> 已按本文约定改为 `注意`，现在 `python scripts/check_gbk_safe.py --strict`
> **退出码 0**。
>
> 💡 **教训**：`check_gbk_safe.py` 不会区分「在 print 里」还是「在 docstring 里」，
> 它是**全字符扫描** —— 这是好事，正好能揪出这种潜伏问题。

---

## 7. 提交红线：这些**不要**入库

已在 `.gitignore` 中排除，列在这里是为了**明确原因**：

| 路径 | 体积 | 为什么不入库 |
|---|---|---|
| `lib/` | 约 335 MB | `pip install` 可随时重建 |
| `linkerhand_sdk/` | 约 1013 MB | 第三方 SDK，来自 Gitee |
| `.venv/` `venv/` `env/` | — | 虚拟环境，机器相关 |
| `robots/*`（**除** `from_teleopbench/`）| 约 274 MB | 代码未引用（`arms/` `assembly/` `h1_paper/` `h1_with_hand/` `hands/` `linker_hand/` `xarm7_ability/`）|
| `tmp_*/` | — | 临时实验目录（如 `tmp_bingdwendwen/`）|
| `data/` `outputs/` `logs/` | — | 运行产物 |
| `_*.log` | — | 脚本调试日志（运行中的进程会占用，删不掉）|
| `*.pt` `*.pth` `*.ckpt` `*.onnx` | — | 模型权重 |
| `*.h5` `*.hdf5` `*.npz`（**`datasets/` 除外**）| — | 采集数据 / 运行产物 |

### 7.1 ⭐ 唯一的例外：`datasets/`

`.gitignore` 里有特殊放行规则：

```gitignore
*.h5                                   # 先全局排除
!datasets/**/*.h5                      # ★ 但 datasets/ 下的必须放行
```

> **为什么必须放行**：契约 G（`human_hand.h5`）与契约 H（`actions.h5`）
> **靠 Git 来交换数据**。如果被 `*.h5` 排除掉，
> 「**队友采集 → 重定向 → 仿真回放**」这条流程会**直接失效**。

### 7.2 `robots/` 的排除写法有讲究

```gitignore
robots/*                      # 先排除所有子目录
!robots/from_teleopbench/     # 再单独放行这一个
```

> ⚠️ **不能**写成 `robots/` ——
> gitignore 规定「**父目录被排除后，无法再重新包含其子文件**」，
> 必须用 `robots/*` + `!robots/from_teleopbench/` 这种写法。

### 7.3 已入库的大资产（唯一一个）

| 路径 | 体积 | 管理方式 |
|---|---|---|
| `robots/from_teleopbench/` | 约 **181 MB** | **普通 Git 对象**（不是 LFS）|

理由见 §3.4：SSH 稳定、队友 `git clone` 一步到位，代价是仓库约 181 MB
（远低于 GitHub 的 1 GB 警告线 / 5 GB 硬限）。

---

## 8. 常用命令速查

### 8.1 环境自检

```powershell
cd F:\simulation_platform
E:\python3.11.7\python.exe test_import.py            # 6 项全 OK
python scripts/check_gbk_safe.py                     # GBK 安全性
python scripts/verify_hand_pipeline.py               # 手部链路 6 项验收
```

### 8.2 Git 日常

```powershell
git -C F:\simulation_platform status
git -C F:\simulation_platform remote -v              # 确认是 SSH 形式
git -C F:\simulation_platform log --oneline -10
git -C F:\simulation_platform fetch --all --prune
git -C F:\simulation_platform push
```

### 8.3 SSH 连通性

```powershell
ssh -T git@github.com                                # 期望 Hi Wangxianyu835!
```

### 8.4 修复 E 盘权限

```powershell
icacls "E:\某目录" /grant "*S-1-5-21-2886930988-4104668371-3581590174-1002:(OI)(CI)(M)" /T
```

### 8.5 检查代理状态

```powershell
Get-Process | Where-Object { $_.ProcessName -match 'clash|mihomo|verge' }
(Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings').ProxyEnable
```

---

## 9. 环境自检清单（换机器 / 重装系统后按顺序做）

| # | 检查项 | 命令 / 标准 | 状态 |
|---|---|---|---|
| 1 | Python 可用 | `E:\python3.11.7\python.exe --version` → 3.11.7 | ☐ |
| 2 | 依赖齐全 | 见 §2.1 验证命令 → 4 个包都打印版本 | ☐ |
| 3 | 项目自检 | `python test_import.py` → 6 项 `[OK]` | ☐ |
| 4 | 模型资产 | `robots\from_teleopbench\` 存在（约 181 MB）| ☐ |
| 5 | SSH 密钥 | `E:\ssh\id_ed25519`（私钥）+ `.pub`（公钥）都在 | ☐ |
| 6 | SSH 连通 | `ssh -T git@github.com` → `Hi Wangxianyu835!` | ☐ |
| 7 | gitconfig | `git config --global --list` 核对 §3.1 各项 | ☐ |
| 8 | 远程 URL | `git remote -v` → **必须是 `git@github.com:...`** | ☐ |
| 9 | 编码检查 | `python scripts/check_gbk_safe.py` → 0 处 | ☐ |
| 10 | E 盘权限 | `icacls E:\ssh` 有 `(OI)(CI)(M)` | ☐ |
| 11 | 手部 SDK（可选）| `linkerhand_sdk\` 存在（只有 `show_hand.py` 需要）| ☐ |
| 12 | **C 盘零写入** | `. .\scripts\env_e_drive_cache.ps1` 后跑 `python scripts\check_no_c_drive.py --strict` → 退出码 0（见 §2.6）| ☐ |

---

## 10. 相关文档与文件

| 文件 | 内容 |
|---|---|
| [`README.md`](../README.md) | 项目介绍、目录结构、使用方法、第三方声明 |
| [`NOTICE`](../NOTICE) | 第三方开源项目的版权归属（TeleOpBench / Unitree 等）|
| [`LICENSE`](../LICENSE) | **Apache-2.0** 完整正文 |
| [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) | 项目背景、技术栈、**决策历史**、与论文的差距清单 |
| [`TEAM_ONBOARDING.md`](TEAM_ONBOARDING.md) | 队友从零上手（约 30 分钟）|
| [`OFFLINE_PIPELINE.md`](OFFLINE_PIPELINE.md) | 离线数据流水线（契约 G / 契约 H）|
| [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) | 模块接口契约 |
| **本文件** | **环境配置与工具链备忘** |

### 给新 AI 对话的开场白

```
请先读这两个文件，再开始工作：
  F:\simulation_platform\docs\PROJECT_CONTEXT.md   （项目背景与决策历史）
  F:\simulation_platform\docs\ENVIRONMENT_SETUP.md （环境配置与坑）

关键前提：项目用 E:\python3.11.7\python.exe 运行（不是项目里的 .venv，那是空的）；
.py 文件禁止出现非 GBK 字符；git 走 SSH，看网页要开代理。
```

---

> **维护提醒**：本文件记录的是**会变的环境状态**。
> 每次解决一个环境类问题（路径变更、权限修复、新踩的坑），
> 就顺手补一节 —— 这类知识最容易被遗忘，也最费时间重新摸索。


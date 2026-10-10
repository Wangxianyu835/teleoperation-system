# 工程约定（PROJECT CONVENTIONS）—— current-state 布局版

> **本项目规则的唯一来源（single source of truth）。**
> 人动手前扫一遍；AI 每次开工前必须读它。**`.clinerules`（仓库根目录）是给 AI 的自动加载版，
> 内容必须与本文件一致** —— 两者冲突时以本文件为准，并立即修 `.clinerules`。
>
> **文件位置**：`F:\simulation_platform_cs\docs\CONVENTIONS.md`
> **最后更新**：2026-10-10（第 1.1 版：基线随上游推进到 `10ebd9f`，`apps/` -> `applications/` 改名与 `tests/` 删除已同步）
> **相关文档**：架构 → [`architecture.md`](architecture.md) · 接口与 H5 契约 → [`contracts.md`](contracts.md) ·
> 上手 → [`README.md`](../README.md) · 旧布局对照见 [`RETARGETING_PIPELINE.md`](RETARGETING_PIPELINE.md) §5

---

## 0. 30 秒版

> **不许写 C 盘 · 用对解释器 · 不改队友代码（基线 `10ebd9f`） · 控制台是 GBK ·
> 改前跑基线、改后复验、结论带退出码。**

每次动手前 / 提交前跑一键自检：

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"                                        # 本机没有 conda 时必需
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1          # 快速 7 项（约 15 秒）
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full    # 再加 unittest + 4 条 CLI 冒烟（约 2 分钟，共 8 项）
```

- 退出码 `0` = 全部通过；`1` = 有 `[FAIL]`，**先修好再动手**（每项都会打印失败原因与修法）。
- 若第 3 项（C 盘）报 `[FAIL]`：说明**这个终端没有重定向过** ——
  先 `. .\devtools\env_e_drive_cache.ps1`（想一劳永逸就加 `-Persist`），再重跑；
  或临时用 `-Fix` 让 preflight 在它自己的进程里重定向（**不改用户级变量**）。

---

## 1. 约定总表

| # | 级别 | 约定 | 违反了会怎样 |
|---|---|---|---|
| **1** | ★★★ | 项目相关的一切**不许写进 C 盘**（依赖 / 缓存 / 临时文件）| 用户硬要求；C 盘空间被吞、返工（2026-10-01 真实翻车：pip 写了 216.78 MB 进 C 盘）|
| **2** | ★★★ | 跑项目**只用正式 conda 环境 `teleoperation`**；本机没有 conda 时回落 `E:\python3.11.7\python.exe` + `PYTHONPATH=<repo>\src` | 用仓库内 `.venv`（空壳）必然 `ModuleNotFoundError: torch`，白排查一整轮 |
| **3** | ★★★ | **不改队友已入库的代码**（基线 `10ebd9f` = `origin/current-state`）| 破坏队友分支的合并与评审（2026-10-01 真实翻车：改了 4 个文件，已全部退回合并点）|
| **4** | ★★★ | `.py` 文件里**禁止非 GBK 字符** | 中文 Windows 控制台会 `UnicodeEncodeError` **崩溃**（不是乱码，是崩溃）|
| **5** | ★★★ | **改前跑基线、改后跑同一套**；结论必须附**命令 + 退出码 + 不变量** | 会得出错误结论并据此改别人的代码（2026-10-01 真实翻车）|
| **6** | ★★★ | 已冻结的技术选型：**PyBullet**（不是 Isaac Sim）、**不用 Git LFS**、数据交换走 **HDF5 契约**（`docs/contracts.md`）| 推翻团队已投入的工作；大文件卡在 GitHub 100 MB 硬限 |
| **7** | ★★★ | **大资产不入库**（`lib/` `.venv/` `outputs/` `logs/` `tmp_*`、模型权重、视频）；`assets/robots/**` 与 `datasets/**` 是**既有例外** | 仓库膨胀、push 失败 |
| **8** | ★★★ | **密钥/凭据**：私钥只放 `E:\ssh`，绝不入库、绝不外传、绝不落到 C 盘 | 安全事故 |
| **9** | ★★ | git 走 **SSH**；**代理只用于看网页 / 开 PR** | `git push` 被 SNI 干扰卡死 |
| **10** | ★★ | 判断脚本成败**看退出码**，别信被管道截断的输出 | 把成功误判成失败（或反之），并据此改错地方 |
| **11** | ★★ | 提交信息 `类型: 说明`，**一个提交一件事**；**中文提交信息写入消息文件** | 历史不可读；PowerShell 引号会破坏中文提交信息 |
| **12** | ★★ | **文档随代码同步更新**；**改约定必须同步两处**（本文件末尾版本表 + `.clinerules`）| 下一个人（或下一个 AI）把同一个坑再踩一遍 |
| **13** | ★★ | **破坏性 / 不可逆操作先说明再执行**（删目录、搬移缓存、`git reset --hard`、改用户级环境变量）| 用户数据丢失且不可恢复 |

> 级别含义：**★★★ 硬约束**（违反 = 返工，没有例外）· **★★ 规范**（违反必须当场说明理由，并写进文档）

---

## 2. 逐条展开

### 约定 1 ★★★ 项目相关的一切不许写进 C 盘

**为什么**：用户明确要求（本机 `C:` 只剩约 20 GB）。这些工具的**默认**缓存/临时目录**全在 C 盘**，
不改环境变量就一定会写进去，且**没有任何报错提示**：

| 工具 | 默认落点 |
|---|---|
| pip 下载缓存 | `C:\Users\<你>\AppData\Local\pip\Cache` |
| HuggingFace（取 URDF、`hf_xet` 日志）| `C:\Users\<你>\.cache\huggingface\` |
| matplotlib 字体/配置 | `C:\Users\<你>\.cache\matplotlib` |
| torch.hub 权重 | `C:\Users\<你>\.cache\torch` |
| 各类 `TEMP` / `TMP` | `C:\Users\<你>\AppData\Local\Temp` |

**怎么守**（每个新终端一次；已 `-Persist` 写进用户级变量则新终端自动生效）：

```powershell
cd F:\simulation_platform_cs
. .\devtools\env_e_drive_cache.ps1              # 7 个变量 -> E:\cache\*（仅当前终端）
E:\python3.11.7\python.exe devtools\check_no_c_drive.py --strict   # 期望 [OK]、退出码 0
```

**怎么查**：`devtools/check_no_c_drive.py --strict`（检查 7 个环境变量 + C 盘 7 处默认缓存位置）；
`devtools/preflight.ps1` 每次都会跑它。

**翻车记录（2026-10-01）**：`pip install -r requirements-retargeting.txt` 把 **216.78 MB / 234 个文件**
写进 `C:\Users\<你>\AppData\Local\pip\Cache`；`hf_xet` 往 `~\.cache\huggingface\xet\logs` 写了日志。

---

### 约定 2 ★★★ 用对解释器

**正式环境（队友声明的基线）**：Conda 环境名 `teleoperation`，Python **3.10.20**，
解释器 `D:\Anaconda\envs\teleoperation\python.exe`，Torch `2.14.1+cu130`（CUDA 可用）。
在该环境里 `python -m teleoperation ...` 直接可用（包以 editable/路径方式已就绪）。

**本机现状（2026-10-10 实测）**：这台机器**没有 Anaconda**（`D:\Anaconda` 不存在），
所以走**回落解释器**：`E:\python3.11.7\python.exe` + **`PYTHONPATH=<repo>\src`**（源码树进 `sys.path`，
包没 pip 安装到它里面）。实测 `python -m teleoperation --help` → 退出码 `0`。

```powershell
cd F:\simulation_platform_cs
$env:PYTHONPATH = "$pwd\src"
E:\python3.11.7\python.exe -m teleoperation --help          # 期望退出码 0
```

**怎么查**：`devtools/preflight.ps1` 第 1 项；有 conda 用它（`[OK]`），没有就 `[WARN]` 并回落（不是失败）。

**不要用**：仓库内 `.venv`（空壳，`python -m teleoperation` 必然 `No module named 'torch'`）、
Anaconda base、`py` 启动器。PATH 上的 `python` 现在是 `F:\simulation_platform\.venv\Scripts\python.exe`
（**旧仓库** `F:\simulation_platform` 的路径；`F:\simulation_platform_cs` 自己不装 `.venv`），
**看到这个就说明你该用全路径**。`devtools/preflight.ps1` 第 1 项对这种情况只报 `[WARN]`
（脚本自己始终用全路径 `$py`，所以不影响退出码 `0`），但**人**别裸敲 `python`。

---

### 约定 3 ★★★ 不改队友已入库的算法代码

**基线**：`10ebd9f`（= `origin/current-state`，队友已合并好的线）。**基线里已存在的**这些路径禁止改：

```
基线 10ebd9f 里全部 747 个已跟踪文件（实测 `(git ls-tree -r --name-only 10ebd9f | Measure-Object).Count` = 747）：
src/teleoperation/**   assets/**   datasets/**   docs/**   pyproject.toml   README.md ...
```

> **基线在 2026-10-10 往前挪过一格**：队友把 `origin/current-state` 从 `42bbe10` 推到
> `10ebd9f`（提交信息"2026-10-9改动：输入文件适配，模型权重更改"），实际内容是
> `src/teleoperation/apps/` -> `src/teleoperation/applications/` **整目录改名**，
> 外加**删除整个 `tests/`（32 个文件）**、删除 `docs/archive/`、`configs/` 里的示例与
> 两个 `datasets/raw/*.h5`（123 files changed / +1301 / -8947）。
> 所以上面清单里**不再有** `tests/**` 与 `configs/**`；本支的路径引用已在
> `chore: 适配上游 10ebd9f 的 apps -> applications 改名` 这一次提交里同步（12 文件 / 49 处）。

`devtools/preflight.ps1` 第 4 项会 `git diff <基线> -- .`，按"文件在基线里是否已存在"判定
（**不是**按目录名），**我新加的文件（`A` 状态）不算违规**；已批准的例外可用 `-Allow <路径>` 放行。

**要兼容怎么办**：新增自己的文件 —— `devtools/**`（自检工具）、`docs/**`、`.clinerules`；
需要挂到队友代码上就写**包装脚本 / 转发层**，不要往他们的文件里插代码。
**队友代码有问题**：写进问题清单（本文件 §3.3）交回队友，自己不要顺手改。

**额外的坑：根目录名字不能乱取**。队友原有的架构测试
`tests/architecture/test_boundaries.py:107`（`test_old_entries_and_path_injection_are_retired`）
把一批**旧布局的根条目名**列为黑名单：`scripts`、`retargeting`、`teleop`、`input_adapters`、
`envs`、`tasks`、`main.py` ... —— 根目录只要出现同名条目，这条用例就 `FAIL`。
（该测试文件**已随上游 `10ebd9f` 删除整个 `tests/` 而消失**；现在这条边界只剩我们自己守：
黑名单写死在 `devtools/preflight.ps1` 第 7 项里，与队友测试是否还在无关。）

**实测教训（2026-10-10）**：我最初把自检工具放在根 `scripts\`，当时让队友这条架构用例多红一次
（cp936 口径 `failures` 6 -> 7；`PYTHONUTF8=1` 口径 0 -> 1）；全部搬到 `devtools\` 后复原。
所以**我们的新增目录一律叫 `devtools/`**。

**实测（2026-10-10，基线 10ebd9f）**：`git diff --name-status 10ebd9f -- .` 共 21 行、
**全部是 `A`**（我这一支新增的 21 个文件：`.clinerules`、`devtools/*` 10 个、
`docs/CONVENTIONS.md`、`docs/RETARGETING_PIPELINE.md`、`src/teleoperation/applications/*` 3 个、
`src/teleoperation/data/capture_h5.py`、`src/teleoperation/retargeting/hand/geometric.py`、
`src/teleoperation/retargeting/arm/wrist_targets.py`、`tests/test_*.py` 2 个），
`0 modified/deleted` 队友文件；架构黑名单 0 命中 → `[OK]`。

---

### 约定 4 ★★★ `.py` 里禁止非 GBK 字符

`✓ ⚠ → ≥ · —` 等字符在中文 Windows 控制台（cp936/GBK）里会让 Python 直接
`UnicodeEncodeError` **崩溃**（不是显示成乱码）。文档（`.md`）不受此限，`.py` 一律用
`[OK]` / `->` / `>=` / `-` 代替。

```powershell
E:\python3.11.7\python.exe devtools\check_gbk_safe.py --strict   # 期望退出码 0、[OK]
E:\python3.11.7\python.exe -m teleoperation tools check-gbk-safe --strict   # 队友自带的等价实现，也要 [OK]
```

队友把同一条规则实现进了 CLI：`tools check-gbk-safe`
（`src/teleoperation/applications/diagnostics/check_gbk_safe.py`，跳过 `lib/`、`robots/`、`linkerhand_sdk/`）。
所以**两条都要过**：他的是出货树的权威检查，我这份确保我们自己的文件也覆盖到
（我的版本额外跳过 `assets/`、`third_party/`、`outputs/`）。

检查范围跳过 `assets/`、`third_party/`、`outputs/`、`lib/`、`.venv/`（不是我们维护的代码）。
**实测（2026-10-10）**：`src/teleoperation` + `tests` 全绿 → `[OK]`。

---

### 约定 5 ★★★ 改前跑基线、改后跑同一套；结论必须附退出码

**两层基线**（改动前后**同一条命令、同一个解释器**各跑一次，比较差异）：

| 层 | 命令 | 判据 |
|---|---|---|
| **冒烟（任何环境都必须过）** | `python -m teleoperation --help` | 退出码 `0` |
| | `python -m teleoperation replay actions --dummy --steps 120 --robot h1_2 --no-render` | 退出码 `0`，无 `Traceback` |
| **全量（改代码/改契约时）** | `python -m unittest discover -s tests -v` | 见下表 |
| | `python -m teleoperation tools check-environment` | 退出码 `0`（6 项，含 300 步无头端到端）|
| | `python -m teleoperation tools verify-hand-pipeline` | 退出码 `0`，末行 `[PASS]` |

`devtools/preflight.ps1 -Full` 会把这四条命令都跑一遍（4 条冒烟也在里面），并顺带跑 GBK 两条检查。

**基线期望值（必须按环境区分，别把本机的既有问题当成自己刚改坏）**：

| 环境 | `unittest discover` 期望 | 说明 |
|---|---|---|
| 正式 conda `teleoperation`（Python 3.10.20）| `OK (skipped=1)` | 队友声明的验收基线；用 `-MaxFail 0 -MaxErr 0` |
| 本机回落 `E:\python3.11.7` + `PYTHONPATH=src` | `Ran 19 tests` / `failures=0` / `errors=0` / `skipped=1` | 建议仍带 `PYTHONUTF8=1`（preflight 自己会设）；判据是**不劣化** |

> 上游 `10ebd9f` 起 `tests/` 目录**整体不存在**（32 个文件被删），所以"队友那 216 项测试"
> 已无从复现；现在 `unittest discover` 跑到的**只有本支的 2 个文件、19 个用例**
> （`tests/test_camera_record.py`、`tests/test_wrist_targets.py`），实测
> `failures=0 / errors=0 / skipped=1`（1 项 skip 是样本 H5 缺失时的守卫，见测试文件顶部说明）。

**为什么以前本机必须开 `PYTHONUTF8=1`**（2026-10-10 实测，**历史记录**）：当时队友的
`tests/test_application_entrypoints.py` 有 6 个子进程用例断言中文结束语"仿真环境已关闭"，
子进程按 UTF-8 打印、父进程按控制台 cp936 解码 → 断言失配。

| 命令 | 结果（旧基线 42bbe10 时代）|
|---|---|
| `python -m unittest discover -s tests -v`（cp936 默认）| `failures=6 / errors=0` |
| `cmd /c "set PYTHONUTF8=1&&python -m unittest discover -s tests -v"` | `failures=0 / errors=0` |

那 6 项是**环境编码问题、不是队友代码缺陷**，修法是设环境变量（**不改队友任何文件**）；
相关用例**已随上游删 `tests/` 而消失**，本支现在的 19 个用例在 cp936 下也全绿。
但"控制台是 GBK"这个事实没变，`devtools/preflight.ps1` 仍在调用 unittest 的那一步自行设置
`PYTHONUTF8=1`、跑完立刻复原。

```powershell
# 本机回落口径：preflight 默认 -MaxFail 0 -MaxErr 17，并自行设置 PYTHONUTF8=1
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full
# conda 正式环境口径：应全绿
powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1 -Full -MaxFail 0 -MaxErr 0
```

**既有失败清单（下面这张表就是 §3.3 引用的那份，**历史记录**）**：在旧基线 `42bbe10` 上，
本机回落的 `unittest` 曾有 24 项不绿，2026-10-10 实测，**全部不是**当时改动引入。
上游 `10ebd9f` 删除整个 `tests/` 后这些项**已无法复现**；表留在这里是为了说明
"不劣化"这条判据当年是怎么定的、那些坑长什么样。

| 数量 | 现象（首行） | 位置 | 性质 |
|---|---|---|---|
| 11 | `NameError: name 'argv' is not defined` | `tests/test_native_hand_replay.py:153` | **队友测试文件的笔误**（应为 `sys.argv`），与解释器无关，100% 复现 → 交回队友 |
| 1 | `AttributeError: 'CanonicalWindowFixture' object has no attribute '_window_or_none'` | `tests/test_hand_core_regression.py:164` 调 `tests/support.py` 的 fixture | **测试夹具与实现不同步**（fixture 缺该方法）→ 交回队友 |
| 2 | `TypeError: Mock.keys() returned a non-iterable (type Mock)` | `src/teleoperation/applications/hand_realtime.py:70` ← `tests/test_mediapipe_realtime.py` 传入未配置的 Mock | 测试桩与实现不兼容（实现要求 `raw.hands`/`raw.metadata` 是 Mapping）→ 交回队友 |
| 3 | `FileNotFoundError: TRON2A URDF was not found: third_party\tron2-robot-description\...` | `tests/test_dual_arm.py` | **外部资产未随仓库提供**：需要设置环境变量 `TRON2A_TEST_URDF` 指向外部 URDF |
| ~~6~~ **0** | `AssertionError: '仿真环境已关闭' not found in ...` | `tests/test_application_entrypoints.py`（子进程用例）| **子进程中文编码 —— 假失败**：设 `PYTHONUTF8=1` 后 6 项全过（见上一张表），队友环境可能本来就是绿的；**不改队友代码** |

**读法**：这些项**都不在**"我改了什么"的范围里 —— 我这一支只**新增**文件
（`devtools/` 10 个、`docs/` 2 个、`.clinerules`、`src/teleoperation/applications/*` 3 个入口、
`src/teleoperation/data/capture_h5.py`、`src/teleoperation/retargeting/{hand/geometric,arm/wrist_targets}.py`、
`tests/test_*.py` 2 个），`git diff --name-status 10ebd9f -- .` 的 21 行**全是 `A`**，没有一处 `M`/`D`。
跑基线要看出的是"**`errors` 与 `failures` 都保持 0**"。

**给用户的结论格式**（禁止只写"已通过"）：

```
改前：<命令> -> 退出码 N（关键数字，如 Ran 19 tests / failures=0 / errors=0 / skipped=1）
改后：<同一命令> -> 退出码 N（关键数字）
差异：只出现/消失了 <具体项>；不变量：<如 verify-hand-pipeline 末行 [PASS]、action_dim=38>
```

**"SKIP 不等于失败"这条经验要留着**：判据永远是**退出码**与**汇总行里的 FAIL/error 数**，
不是"PASS 数等于某个写死的数字"。旧布局里 `run_retargeting_pipeline.py` 第 0 步复用已有 raw 时打 `[SKIP]`
（汇总 `PASS=6 FAIL=0 SKIP=1` 也是成功），新布局里对应的是 unittest 的 `skipped=1`（外部 URDF 缺失的用例）
—— `OK (skipped=1)` = 通过。

---

### 约定 6 ★★★ 技术选型已冻结

- 仿真用 **PyBullet**（不是 Isaac Sim / MuJoCo）；机器人 USD/MJCF 不引入。
- 版本管理用**普通 Git + SSH**，**不用 Git LFS**（`origin/current-state` 的 `.gitattributes` 已明确写"不使用 Git LFS"）。
- 数据交换走 **HDF5 契约**，以 `docs/contracts.md` 为准；不要私改字段名/结构。
- 要改这三项 → **先问用户**。

### 约定 7 ★★★ 大资产不入库

**禁止入库**：`lib/`、`linkerhand_sdk/`、`.venv/`、`outputs/`、`logs/`、`tmp_*`、
模型权重（`.pt/.pth/.ckpt/.onnx`）、视频（`.mp4/.avi`）、非 datasets 的 `.h5/.hdf5`。

**既有例外（基线里就跟踪着，别自作主张删）**：
- `assets/robots/**` —— 正式机器人资源（URDF/STL/网格）随仓库管理；
- `datasets/**` —— H5 契约样本与录制（`.gitignore` 里用 `!datasets/**/*.h5` 放行）。

**临时产物放哪**：`outputs/` 整目录已被 `.gitignore` 忽略 → 临时/实验文件写
`outputs\tmp_<用途>\`，**不要在仓库根目录建 `tmp_*`**（cs 的 `.gitignore` 没有 `tmp_*` 规则，
根目录下的 `tmp_*` 会变成未跟踪文件，preflight 第 5 项会报 FAIL）。

**怎么查**：`preflight.ps1` 第 5 项扫 `git status --porcelain`：命中禁止前缀、或暂存/未跟踪的
> 5 MB 且不属于例外的文件 → `[FAIL]`。

---

### 约定 8 ★★★ 密钥 / 凭据

私钥、Token、SID 只放 `E:\ssh`（已建），**绝不入库、绝不外传、绝不落到 C 盘**；
文档、提交信息、聊天记录里都不写实际内容（要写就写"见 `E:\ssh\<文件名>`"）。
`.gitignore` 已有私钥模式；新增密钥文件后自己 `git status` 复核一遍没被跟踪。

### 约定 9 ★★ git 走 SSH，代理只用于浏览网页

`git remote` 用 `ssh://git@github.com/...` 或 `git@github.com:...`；
网页访问 / 开 PR 才用代理（本机代理常驻会干扰 `git push` 的 SNI，表现为卡死不报错）。

### 约定 10 ★★ 判断脚本成败看退出码

- PowerShell 里看 `$LASTEXITCODE`；**不要**看被 `Select-Object -First N` 截断的管道输出。
- 最稳写法：`cmd /c "E:\python3.11.7\python.exe -m teleoperation ... > outputs\tmp_x\log 2>&1"`，
  然后只判退出码 + 抓汇总行。
- **不是错误**：PyBullet 往 stderr 打的 `pybullet build time: ...`、`b3Warning[...]`（缺惯量数据等）；
  `--no-render` 模式下 `[结果] {'success': False, ...}` 只表示"这一小节任务没完成"，不是崩溃。
- 中文输出在重定向日志里可能显示成乱码（cp936/UTF-8 混合）：**先看退出码**，再决定要不要深挖日志。

### 约定 11 ★★ 提交信息

- 格式 `类型: 说明`，类型取 `feat / fix / docs / chore / data / test / refactor`，**一个提交一件事**。
- **中文提交信息写进消息文件**再用 `git commit -F`；直接写在命令行里会被 PowerShell 引号弄坏：

```powershell
# 先用编辑器把信息写进 outputs\tmp_commit\msg.txt（UTF-8），再：
git add devtools docs .clinerules
git commit -F outputs\tmp_commit\msg.txt
```

- 提交前跑 `devtools/preflight.ps1`（至少快速 7 项），提交后 `git log --oneline -1` 复核中文没乱码。

---

### 约定 12 ★★ 文档随代码同步更新；改约定必须同步两处

**改约定时**（两处必须一致，缺一处等于没改）：

1. 本文件 `docs/CONVENTIONS.md`（正文 + 末尾「版本」表加一行）
2. `.clinerules`（仓库根目录，**给 AI 的自动加载版**）

> 旧布局里第 3 处是 `docs/PROJECT_CONTEXT.md` 的 §0.1；那份手册队友先归档到 `docs/archive/`，
> 随后又在上游 `10ebd9f` 里**连同 `docs/archive/` 整个目录一起删除**，所以它在本仓库里
> 已经不存在了，本分支不再要求同步它 —— 避免与队友文档打架。
> 若将来恢复一份"活跃手册"，再加回第 3 处。

**代码/工具改动时**：更新对应文档（如 `docs/RETARGETING_PIPELINE.md`、本文件的实测记录），
并改文首「最后更新」日期。改约定本身：先问用户。

### 约定 13 ★★ 破坏性 / 不可逆操作先说明再执行

包括但不限于：`Remove-Item -Recurse`、`robocopy /MOVE`、`git reset --hard`、`git clean -fd`、
`git push --force`、`git worktree remove`、修改**用户级**环境变量、安装/卸载系统组件。

**怎么守**：动手前用一句话说清"要删/移什么、多少量、能否恢复、怎么恢复"；
**能搬不删**（`/MOVE` 优于删除）；先备份到 `outputs\tmp_*\` 再操作。

---

## 3. 基线与验收口径（`devtools/preflight.ps1` 的 8 项）

### 3.1 检查项一览

| # | 检查 | 依据 | 失败时怎么办 |
|---|---|---|---|
| 1 | 解释器（conda `teleoperation` 优先，回落 `E:\python3.11.7\python.exe` + `PYTHONPATH=src`；PATH 上的裸 `python` 若指向 `.venv` 只报 `[WARN]`）| 约定 2 | 装好 conda，或用回落解释器并注明；命令一律写全路径 |
| 2 | `.py` 非 GBK 字符（`devtools/check_gbk_safe.py --strict`）| 约定 4 | 把 `✓ → ≥ ⚠` 换成 `[OK] -> >= !` |
| 2b | 同一规则走队友 CLI：`python -m teleoperation tools check-gbk-safe --strict` | 约定 4 | 同上（他这份是出货树的权威检查）|
| 3 | C 盘零写（`devtools/check_no_c_drive.py --strict`：7 个环境变量 + 7 处默认缓存位置）| 约定 1 | `. .\devtools\env_e_drive_cache.ps1 -Persist`；或 `preflight.ps1 -Fix`（只改当前进程）|
| 4 | 队友已入库文件零改动（`git diff <基线> -- .`；`A` 状态放行）| 约定 3 | `git checkout <基线> -- <文件>`，逻辑搬进自己的新文件；确需例外用 `-Allow <路径>` |
| 5 | 没有禁入库目录/大资产进暂存（`lib/`、`.venv/`、`outputs/`、`tmp_*`，或 `.pt/.h5/...` > 5 MB）| 约定 7 | `git restore --staged <路径>`，文件搬进 `outputs/tmp_*/` |
| 6 | 规则文件在位（`.clinerules`、`docs/CONVENTIONS.md`）| 约定 12 | 补回文件 |
| 7 | 架构边界：11 个旧布局根条目名（`scripts`、`retargeting`、`teleop`...）+ `src/teleoperation/**` 里的 `sys.path.insert` | 原属队友 `tests/architecture/test_boundaries.py:107`（上游 `10ebd9f` 已删该文件），现由本脚本自查 | 改名/搬家（我们的工具目录固定叫 `devtools/`）|
| 8 | `-Full`：`unittest discover`（判 `failures<=-MaxFail`、`errors<=-MaxErr`）+ 4 条 CLI 冒烟 | 约定 5 | 当前基线实测 `19 tests / failures=0 / errors=0 / skipped=1`（§3.2）|

退出码：`0` = 无 `[FAIL]`（`[WARN]` 允许），`1` = 至少一项 `[FAIL]`。
开关：`-Full`、`-Base <sha>`（默认 `10ebd9f`）、`-MaxFail N`（默认 `0`）、`-MaxErr N`（默认 `17`，宽松上限；基线实测 `errors=0`）、
`-Allow <路径>`、`-Fix`。加严口径：`-MaxFail 0 -MaxErr 0`。

### 3.2 解释器与期望值（含 `PYTHONUTF8=1`）

| 环境 | `python -m unittest discover -s tests -v` 期望（2026-10-10 实测）| `preflight.ps1 -Full` 参数 |
|---|---|---|
| 正式 conda `teleoperation`（Python 3.10.20）| `OK (skipped=1)` | 默认即可；更严用 `-MaxFail 0 -MaxErr 0` |
| 本机回落 `E:\python3.11.7\python.exe` + `PYTHONPATH=src` | `Ran 19 tests` / `failures=0` / `errors=0` / `skipped=1` | 默认（`-MaxFail 0 -MaxErr 17`），**建议仍带 `PYTHONUTF8=1`**（上游删 `tests/` 前不带它会多 6 项假失败）|

**`PYTHONUTF8=1` 是怎么变成"硬要求"的**（历史）：上游删 `tests/` 之前，那 6 个子进程用例
断言中文结束语"仿真环境已关闭"，子进程按 UTF-8 打印、父进程按控制台 cp936 解码 → 断言失败
（两张对照表见约定 5）。那些用例现在**不在仓库里了**，但"控制台是 GBK"这个事实没变，
所以 `devtools/preflight.ps1` 仍在跑 unittest 前自行设置、跑完立刻复原；手工跑用：

```powershell
cmd /c "set PYTHONUTF8=1&&E:\python3.11.7\python.exe -m unittest discover -s tests -v"
```

注意 `set VAR=1&&cmd` 里 **`&&` 前不能有空格**（`set VAR=1 &&cmd` 会把尾随空格算进值里，
Python 报 `invalid PYTHONUTF8 value`）—— 这也是约定 10"看退出码、别信截断管道"的又一例证。

### 3.3 既有问题清单（交回队友，别自己改）

**当前没有既有问题**：实测 `errors=0 / failures=0`（19 个用例，2026-10-10）。
下面这段是**历史归档**：旧基线 `42bbe10` 时代曾有 17 个 error，**全部**是队友测试侧的问题
（逐项明细见约定 5 的那张表）：**队友测试笔误（11 项 `argv`）**、**测试夹具与实现不同步（1 项）**、
**测试桩与实现不兼容（2 项）**、**外部资产未随仓库提供（3 项 TRON2A URDF）**。
上游 `10ebd9f` 删除整个 `tests/` 后这些项**不再复现**，这张清单只作"当年的结论"保留。

已经查清的**假失败**（结论：**不需要动队友任何文件**）：

- 6 项 `'仿真环境已关闭' not found`：cp936 解码 UTF-8 输出所致 → 设 `PYTHONUTF8=1` 即全部消失（§3.2）。
- 1 项架构用例一度变红：根目录出现 `scripts/` 触发队友黑名单（§3.1 第 7 项）→
  工具搬到 `devtools/` 后复原（`failures` 7 → 0）。

**判据**：`errors` 保持 0、`failures` 保持 0、4 条冒烟退出码 0。**不要求"全绿"**，要求**不劣化**。
（`-MaxErr` 默认仍是 `17`，作为"上游把测试套件补回来"时的宽松上限；想收紧就用 `-MaxErr 0`。）

---

## 4. 日常工作流（从开终端到提交）

1. **开终端**：`. .\devtools\env_e_drive_cache.ps1`（已 `-Persist` 过则新终端自动生效）→
   `powershell -ExecutionPolicy Bypass -File devtools\preflight.ps1`（快速 7 项，期望 `RESULT: [OK]`）。
2. **改前跑基线**：`devtools\preflight.ps1 -Full`，把汇总行抄下来
   （`Ran 19 tests` / `failures=0` / `errors=0` / `skipped=1` + 4 条冒烟 `exit 0`）。
3. **动手**：只**新增**自己的文件（`devtools/**`、`docs/**`、`.clinerules`）；队友文件只读；
   根目录不要新建 `scripts/`、`retargeting/` 之类名字（§3.1 第 7 项）。
4. **改后跑同一套**：同命令、同解释器再跑一次 `-Full`，按约定 5 的格式写差异。
5. **提交前**：`devtools\check_gbk_safe.py --strict` 与 `-m teleoperation tools check-gbk-safe --strict`
   都退出码 0；`git status --untracked-files=all` 确认没把大资产/临时文件带进来。
6. **提交**：中文信息写进消息文件 → `git add devtools docs .clinerules` → `git commit -F <消息文件>`。
7. **汇报**：命令 + 退出码 + 关键不变量（约定 5 的格式），结论先行。

---

## 5. 常见问题 FAQ

| 症状 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'teleoperation'` | 没把 `src` 放进 `sys.path` | `$env:PYTHONPATH="$pwd\src"`（preflight 会替你设）|
| `No module named 'torch'` | 用了仓库内 `.venv`（空壳）| 用完整路径的解释器（约定 2）|
| preflight 第 3 项 `[FAIL]` | 这个终端没重定向缓存 | `. .\devtools\env_e_drive_cache.ps1 -Persist`，或 `-Fix` |
| unittest 出现 `'仿真环境已关闭' not found` | 没设 `PYTHONUTF8=1`（该用例已随上游删 `tests/` 消失）| 见 §3.2，属假失败 |
| 旧架构用例 `test_old_entries_and_path_injection_are_retired` 变红（**上游已删该文件**）| 根目录出现了黑名单名字 | 改名/搬家（我们的工具目录固定用 `devtools/`）；`preflight.ps1` 第 7 项同样会拦 |
| 日志里中文是乱码 | cp936 控制台读 UTF-8 输出 | 先看退出码，再决定要不要深挖（约定 10）|
| 找不到 conda 环境 `teleoperation` | 本机没装 | 用回落解释器口径（§3.2），汇报时注明 |

---

## 6. 版本

| 日期 | 版本 | 变更 | 影响文件 |
|---|---|---|---|
| 2026-10-10 | v1.0 | 首版：旧布局（`retargeting/` 等）的约定移植到 `current-state` 布局；新增 `devtools/` 工具链与 `docs/RETARGETING_PIPELINE.md`；确认两个关键结论 —— 必须 `PYTHONUTF8=1`、工具目录必须叫 `devtools/`（队友架构黑名单）| `docs/CONVENTIONS.md`、`.clinerules`、`docs/RETARGETING_PIPELINE.md`、`devtools/*` |
| 2026-10-10 | v1.1 | **基线随上游推进到 `10ebd9f`**（`src/teleoperation/apps/` -> `applications/` 整目录改名；上游删除整个 `tests/`、`docs/archive/`、`configs/` 与两个 `datasets/raw/*.h5`）：本支 rebase 到新基线、同步 49 处路径引用，并把所有钉死的数字刷新（基线 `747` 文件 / `Ran 19 tests` / `errors=0`）；`preflight.ps1` 的 `-Base` 默认值由 `42bbe10` 改为 `10ebd9f` | `docs/CONVENTIONS.md`、`.clinerules`、`docs/RETARGETING_PIPELINE.md`、`devtools/*` |

> 改约定 = 本表加一行 + `.clinerules` 同步（约定 12）。

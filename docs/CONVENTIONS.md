# 工程约定（PROJECT CONVENTIONS）

> **本项目规则的唯一来源（single source of truth）。**
> 人动手前扫一遍；AI 每次开工前必须读它。**`.clinerules`（仓库根目录）是给 AI 的自动加载版，
> 内容必须与本文件一致** —— 两者冲突时以本文件为准，并立即修 `.clinerules`。
>
> **文件位置**：`F:\simulation_platform\docs\CONVENTIONS.md`
> **最后更新**：2026-10-01（第 1 版）
> **相关文档**：背景与历史 → [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)（§0.1 硬约束摘要、§8.1 时间线）·
> 环境细节 → [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md)（§2.6 C 盘、§6 GBK）·
> 队友上手 → [`TEAM_ONBOARDING.md`](TEAM_ONBOARDING.md)

---

## 0. 30 秒版

> **不许写 C 盘 · 只用 E 盘的解释器 · 不改队友代码 · 控制台是 GBK ·
> 改前跑基线、改后复验、结论带退出码。**

每次动手前 / 提交前跑一键自检：

```powershell
cd F:\simulation_platform
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1          # 快速 6 项（约 10 秒）
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Full    # 再加 pytest + 重定向流水线（约 2 分钟）
```

- 退出码 `0` = 全部通过；`1` = 有 `[FAIL]`，**先修好再动手**（每项都会打印失败原因与修法）。
- 若第 3 项（C 盘）报 `[FAIL]`：说明**这个终端没有重定向过** ——
  先 `. .\scripts\env_e_drive_cache.ps1`（想一劳永逸就加 `-Persist`），再重跑；
  或临时用 `-Fix` 让 preflight 在它自己的进程里重定向（**不改用户级变量**）。

---

## 1. 约定总表

| # | 级别 | 约定 | 违反了会怎样 |
|---|---|---|---|
| **1** | ★★★ | 项目相关的一切**不许写进 C 盘**（依赖 / 缓存 / 临时文件）| 用户硬要求；C 盘空间被吞、返工（2026-10-01 真实翻车：pip 写了 216.78 MB 进 C 盘）|
| **2** | ★★★ | 跑项目**只用 `E:\python3.11.7\python.exe`** | 用仓库内 `.venv`（空壳）必然 `ModuleNotFoundError: torch`，白排查一整轮 |
| **3** | ★★★ | **不改队友已入库的算法代码** | 破坏队友分支的合并与评审（2026-10-01 真实翻车：改了 4 个文件，已全部退回合并点）|
| **4** | ★★★ | `.py` 文件里**禁止非 GBK 字符** | 中文 Windows 控制台会 `UnicodeEncodeError` **崩溃**（不是乱码，是崩溃）|
| **5** | ★★★ | **改前跑基线、改后跑同一套**；结论必须附**退出码 + 不变量** | 会得出错误结论并据此改别人的代码（2026-10-01 真实翻车）|
| **6** | ★★★ | 已冻结的技术选型：**PyBullet**（不是 Isaac Sim）、**不用 Git LFS**、数据交换走 **HDF5 契约 G/H** | 推翻团队已投入的工作；大文件卡在 GitHub 100 MB 硬限 |
| **7** | ★★★ | **大资产不入库**（`lib/` `linkerhand_sdk/` `.venv/` `tmp_*/`、模型权重、视频）| 仓库膨胀、push 失败 |
| **8** | ★★★ | **密钥/凭据**：私钥只放 `E:\ssh`，绝不入库、绝不外传、绝不落到 C 盘 | 安全事故 |
| **9** | ★★ | git 走 **SSH**；**代理只用于看网页 / 开 PR** | `git push` 被 SNI 干扰卡死 |
| **10** | ★★ | 判断脚本成败**看退出码**，别信被管道截断的输出 | 把成功误判成失败（或反之），并据此改错地方 |
| **11** | ★★ | 提交信息 `类型: 说明`，**一个提交一件事**；**中文提交信息写入消息文件** | 历史不可读；PowerShell 引号会破坏中文提交信息 |
| **12** | ★★ | **文档随代码同步更新**；**改约定必须同步三处**（本文件 / PROJECT_CONTEXT §0.1 / `.clinerules`）| 下一个人（或下一个 AI）把同一个坑再踩一遍 |
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
| HuggingFace（`robot_descriptions` 取 URDF、`hf_xet` 日志）| `C:\Users\<你>\.cache\huggingface\` |
| matplotlib 字体/配置 | `C:\Users\<你>\.cache\matplotlib` |
| torch.hub 权重 | `C:\Users\<你>\.cache\torch` |
| 各类 `TEMP` / `TMP` | `C:\Users\<你>\AppData\Local\Temp` |

**怎么守**（每个新终端一次；已 `-Persist` 写进用户级变量则新终端自动生效）：

```powershell
cd F:\simulation_platform
. .\scripts\env_e_drive_cache.ps1              # 7 个变量 -> E:\cache\*（仅当前终端）
E:\python3.11.7\python.exe scripts\check_no_c_drive.py --strict   # 期望 [OK]、退出码 0
```

**怎么查**：`scripts/check_no_c_drive.py --strict`（检查 7 个环境变量 + C 盘 7 处默认缓存位置）；
`scripts/preflight.ps1` 每次都会跑它。

**翻车记录（2026-10-01）**：`pip install -r requirements-retargeting.txt` 把 **216.78 MB / 234 个文件**
写进 `C:\Users\王宪雨\AppData\Local\pip\Cache`；`hf_xet` 往 `~\.cache\huggingface\xet\logs` 写了日志。
已建 `E:\cache\{pip,matplotlib,huggingface,torch,xdg,tmp}`、把已有 **1560.54 MB / 889 个文件**
pip 缓存 `robocopy /MOVE` 到 `E:\cache\pip`（缓存仍可复用），并把规则写进 §0.1 硬约束与 README。
细节与完整修复步骤见 [`ENVIRONMENT_SETUP.md`](ENVIRONMENT_SETUP.md) §2.6。

> ⚠️ **已知无害残留**：IDE（VS Code + 扩展）在命令输出过大时会写
> `C:\Users\<你>\AppData\Local\Temp\cline\*.log`（每个几 KB）—— 这是扩展自身行为，环境变量管不到它。
> 应对：**不要让单条命令刷出巨量输出**，定期清该目录。

**违反后果**：返工 + 向用户解释；已写进 C 盘的东西必须**搬走**（不是删掉，见约定 13）。

---

### 约定 2 ★★★ 跑项目只用 `E:\python3.11.7\python.exe`

**为什么**：仓库里的 `.venv` 是**空壳**（只有 `pip / setuptools / Pillow / pypdf`），
而终端提示符偏偏显示 `(.venv)`，非常有迷惑性。用它跑项目必然 `No module named 'torch' / 'pybullet'`。

**怎么守**：命令里**显式写全路径**，不要依赖 `python` 的 PATH 解析。

```powershell
E:\python3.11.7\python.exe -c "import sys; print(sys.executable)"   # 必须打印 E:\python3.11.7\python.exe
E:\python3.11.7\python.exe -m pytest tests -q
E:\python3.11.7\python.exe main.py --task pushcube
```

**怎么查**：`scripts/preflight.ps1` 第 1 项 —— 打印解释器版本，并列出 PATH 上的 `python` 指向（若不同则告警）。

**违反后果**：白排查一整轮（2026-10-01 就因为空壳 `.venv` 报错，误判成"环境缺依赖会连累整仓"，
进而改了队友的 `retargeting/arm.py` —— 见约定 3）。

---

### 约定 3 ★★★ 不改队友已入库的算法代码

**为什么**：`retargeting/` 是队友分支的产物，改了他那边的合并会冲突、评审记录也会失真；
"顺手方便一下"的改动最后往往被证明**根本不需要**（本项目已经发生过一次）。

**范围**（以合并点 `3e4e763` 里**已存在**的文件为准）：

```
retargeting/**      arm.py tracking.py inference.py training.py data.py model.py config.py
                    inspect.py dual_teleop.py simulation.py visionpro.py mediapipe.py
                    command.py coordinates.py hand_core.py realtime_dual_teleop.py
                    contracts.py __init__.py __main__.py
config/**           retarget_io.py + tron2a_dach_calibration.example.json
input_adapters/**   除自己新增的文件外
tests/**            除自己新增的文件外
inspect_angle_h5.py
main_offline_dual_teleop.py
```

**怎么守**（三条路，都不动队友的文件）：

| 需求 | 正确做法 |
|---|---|
| 缺常量 / 缺接口 | **自己新增文件**做转发层或包装（如 `input_adapters/hand_keypoints.py` 用 try-import 兜底：队友日后自己补上同名常量即自动以其为准）|
| 测试因缺资产（URDF 等）需跳过 | **自己新增 `tests/conftest.py`** 钩子，不改 `tests/test_dual_arm.py` |
| 队友代码本身有问题 | **不改**，写进问题清单交回队友（[`RETARGETING_PIPELINE.md`](RETARGETING_PIPELINE.md) §9.1），附复现命令与实测证据 |

**怎么查**：`scripts/preflight.ps1` 第 3 项 —— 把"队友侧路径"里**在合并点就存在**的文件与当前工作区做
`git diff`，任何 `M` / `D` / `R` 都算 FAIL（**自己新增的文件不算**）。基线可用 `-Base <commit>` 覆盖。

**翻车记录（2026-10-01）**：曾改 `retargeting/arm.py`、`retargeting/tracking.py`、`tests/test_dual_arm.py`、
`inspect_angle_h5.py` 共 4 个文件，理由（"缺 `roboticstoolbox` 会连累整仓"）事后被实测推翻 ——
已 `git checkout 3e4e763` 全部还原，兼容逻辑移入自建文件。

---

### 约定 4 ★★★ `.py` 文件里禁止非 GBK 字符

**为什么**：中文 Windows 控制台是 **cp936(GBK)**。代码里出现 `✓ ✗ ⚠ → ≥ ²` 这类字符并 `print()` 时，
Python 抛 `UnicodeEncodeError` **直接崩溃**（不是显示乱码）。`.md` 文档不受此限。

**怎么守**：用 ASCII 替代表达 —— `[OK]` / `[FAIL]` / `注意` / `->` / `>=` / `^2`。

```powershell
E:\python3.11.7\python.exe scripts\check_gbk_safe.py --strict   # 期望退出码 0
```

**怎么查**：`scripts/preflight.ps1` 第 2 项（会给出具体 `文件:行号` 与建议写法）。

**违反后果**：程序崩溃。这类字符曾在 `teleop/native_hand.py` 的 docstring 里潜伏很久。

---

### 约定 5 ★★★ 改前跑基线、改后跑同一套；结论必须附退出码与不变量

**为什么**："看起来对"是本项目最大的风险源 —— 一次基于错误前提的改动，代价是改坏了别人的代码。

**怎么守**：

1. **改前**：`scripts/preflight.ps1 -Full`，把 `pytest tests -q` 与重定向流水线的结果**记下来**
   （例如 `40 passed, 3 skipped`、`7/7 PASS`）；
2. 动手改；
3. **改后**：跑**同一套**命令，逐项对比；
4. 交付结论必须同时给出：**命令** + **退出码** + **关键不变量**（数量 / 维度 / 误差等），不许只写"已通过"。

```powershell
cd F:\simulation_platform
$py = 'E:\python3.11.7\python.exe'
. .\scripts\env_e_drive_cache.ps1
& $py -m pytest tests -q;                  "PYTEST_EXIT=$LASTEXITCODE"
& $py scripts\run_retargeting_pipeline.py;  "PIPELINE_EXIT=$LASTEXITCODE"
```

**违反后果**：得出错误结论（本轮真实翻车：由"`.venv` 报错"推出"缺依赖会连累整仓"，
据此改了队友 `arm.py`，实测完全不必要、全部回退）。

---

### 约定 6 ★★★ 已冻结的技术选型（不要擅自推翻）

| 冻结项 | 内容 | 出处 |
|---|---|---|
| 仿真器 | **PyBullet**，不是 Isaac Sim（用户 2026-09-12 明确"我就是要用 PyBullet 做"）| PROJECT_CONTEXT §5 / §13 |
| 版本管理 | **普通 Git + SSH，不用 Git LFS**（LFS 强制 HTTPS，国内网络会被干扰）| TEAM_ONBOARDING §6 |
| 数据交换 | **HDF5 契约 A~H**（`datasets/` 下 `human_hand.h5` / `actions.h5` 靠 git 交换）| INTERFACE_CONTRACT |
| 阶段范围 | 先做**手部**链路、暂不含手臂联动 | OFFLINE_PIPELINE §8.9 |

**怎么守**：要改这些，**先问用户**；同意后更新本表 + PROJECT_CONTEXT 对应章节，并在提交信息里写明理由。

---

### 约定 7 ★★★ 大资产不入库

| 不入库 | 体积 | 获取方式 |
|---|---|---|
| `lib/` | ~335 MB | `pip install -r requirements.txt -t lib` |
| `linkerhand_sdk/` | ~1013 MB | `git clone https://gitee.com/ericbrunt/linkerhand_telop_python.git linkerhand_sdk` |
| `.venv/` | — | 不需要（见约定 2）|
| `tmp_*/`（临时实验目录、调试日志）| — | 就地生成，`.gitignore` 已排除 |
| 模型权重 `*.pt/.pth/.ckpt/.onnx`、视频 `*.mp4/.avi`、`data/` `outputs/` `logs/` | — | 移动硬盘 / 网盘 |

**例外（确实入库）**：`robots/from_teleopbench/**`（181 MB，clone 即得）与 `datasets/**/*.h5`（契约 G/H 数据）。

**怎么查**：`scripts/preflight.ps1` 第 4 项 —— 扫描 `git status` 里是否混入上述路径，
以及是否有超过 5 MB 的新文件（`robots/`、`datasets/` 除外）。

---

### 约定 8 ★★★ 密钥与凭据

- 私钥只放 `E:\ssh\id_ed25519`（**本来就不许放 C 盘**，用户要求；并已收紧 ACL）。
- **绝不** `git add` 任何私钥 / token / 凭据；文档里**绝不**写密钥、密码、Token、SID 的实际内容。
- 提交前自检：

```powershell
git diff --cached --name-only | Select-String 'id_ed25519|\.pem$|token|credential|secret'
# 期望：没有任何输出
```

---

### 约定 9 ★★ git 走 SSH；代理只用于看网页 / 开 PR

```powershell
git remote -v        # 必须是 git@github.com:...（不能是 https://...）
git config --global core.sshCommand "C:/Windows/System32/OpenSSH/ssh.exe"   # 中文用户名必须显式指定
ssh -T git@github.com                                                      # 期望 Hi Wangxianyu835!
```

看 GitHub 网页 / 点 Merge 时才开代理（选住宅节点）。详见 ENVIRONMENT_SETUP §3~§4。

---

### 约定 10 ★★ 判断脚本成败看退出码

**坑（本项目实测）**：

- PowerShell 的 `| Select-Object -Last 20` 会提前关掉管道上游，导致脚本**被判失败**（退出码 1），
  而输出看起来是完整的；
- pybullet 会往 **stderr** 打 `pybullet build time: ...`，PowerShell 把它标红成 `NativeCommandError`
  —— **这不是错误**（同类：`b3Warning: No inertial data for link`）。

**怎么守**：

```powershell
& $py scripts\xxx.py > tmp_motion\xxx.log 2>&1   # 1) 落盘（tmp_* 不入库）
"EXIT=$LASTEXITCODE"                              # 2) 只看退出码
Get-Content tmp_motion\xxx.log -Tail 30           # 3) 再看尾部输出
```

或直接 `cmd /c "... > log 2>&1"` 绕开 PowerShell 的 stderr 语义（本项目实测最稳）。

---

### 约定 11 ★★ 提交信息与提交粒度

- 格式：`类型: 说明`，类型取 `feat` / `fix` / `docs` / `chore` / `data` / `test` / `refactor`，可带 `(scope)`。
- **一个提交一件事**，不要把无关改动混进同一个提交（方便回滚与评审）。
- **中文提交信息必须用消息文件**（PowerShell 的引号会把中文弄乱或截断）：

```powershell
# 消息先写进 tmp_motion\commit_msg.txt，然后：
git commit -F tmp_motion\commit_msg.txt
```

- 文档类改动可直接提交 `main`；功能 / 修复类改动走分支 + PR（见 PROJECT_CONTEXT §11.3）。

---

### 约定 12 ★★ 文档随代码同步更新；改约定必须同步三处

**改约定时**（三处必须一致，缺一处等于没改）：

1. 本文件 `docs/CONVENTIONS.md`（正文 + 末尾版本表加一行）
2. `docs/PROJECT_CONTEXT.md` §0.1 硬约束表 + §8.1 时间线追加一行
3. `.clinerules`（仓库根目录，**给 AI 的自动加载版**）

**代码改动时**：按 PROJECT_CONTEXT §11.1 的更新清单同步相关章节，并改文首"最后更新"日期。
**改约定本身**：先问用户，提交信息用 `docs(conventions): ...`。

---

### 约定 13 ★★ 破坏性 / 不可逆操作先说明再执行

包括但不限于：`Remove-Item -Recurse`、`robocopy /MOVE`、`git reset --hard`、`git clean -fd`、
`git push --force`、修改**用户级**环境变量、安装/卸载系统组件。

**怎么守**：动手前用一句话说清"要删/移什么、多少量、能否恢复、怎么恢复"；
**能搬不删**（`/MOVE` 优于删除）；先备份到 `tmp_*/` 再操作。

---

## 3. 一键自检：`scripts/preflight.ps1`

```powershell
cd F:\simulation_platform
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1                 # 快速 6 项（约 10 秒）
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Full           # 再加 pytest + 重定向流水线
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Fix            # 在自己进程里先重定向缓存变量（不写用户级变量）
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1 -Base 3e4e763   # 指定「队友代码基线」提交
```

| 项 | 检查内容 | 对应约定 |
|---|---|---|
| 1 | 解释器是 `E:\python3.11.7\python.exe`（顺带打印 PATH 上的 `python`）| 约定 2 |
| 2 | `check_gbk_safe.py --strict`（`.py` 里无非 GBK 字符）| 约定 4 |
| 3 | `check_no_c_drive.py --strict`（缓存/临时目录已重定向、C 盘无残留）| 约定 1 |
| 4 | 队友已入库代码零改动（`git diff <基线>`，只算**基线里已存在**的文件）| 约定 3 |
| 5 | 没有把 `lib/ linkerhand_sdk/ .venv/ data/ outputs/ logs/` 或大资产（>5 MB / 权重 / 视频 / h5）加进暂存区 | 约定 7 |
| 6 | `.clinerules` 与 `docs/CONVENTIONS.md` 都在 | 约定 12 |
| 7 | `-Full` 时：`pytest tests -q`（期望 `40 passed, 3 skipped`）+ `run_retargeting_pipeline.py`（期望 `7/7 PASS`）| 约定 5 |

**输出约定**：每项一行 `[OK  ] / [WARN] / [FAIL]`，最后一行是 `RESULT: [OK] ...` 或 `RESULT: [FAIL] ...`；
有 `FAIL` 时退出码 `1`。脚本自身的输出**全是 ASCII**（PowerShell 5.1 在 GBK 控制台读 `.ps1` 会把中文字面量弄坏），
子检查器是 Python、会打印中文。

**实测（2026-10-01）**：正常路径 `RESULT: [OK]`、退出码 `0`；
故意在 `retargeting/arm.py` 末尾加一行 → 第 4 项抓到 `M retargeting/arm.py` 并给出还原命令、退出码 `1`；
`-Full` 下 pytest `40 passed, 3 skipped`、流水线 `7/7 PASS`。

---

## 4. 约定怎么演进

- 约定不是刻在石头上的，但**改它要走三处同步 + 先问用户**（见约定 12）。
- 新踩一个坑 → 判断它属于哪一条：若是新类型，**加一条约定**并配一个可执行的检查；若是旧坑复发，
  说明现有检查没拦住，**升级检查**（能自动查的绝不靠提醒）。
- 任何约定都必须满足：**可执行**（有命令）、**可判定**（有 OK/FAIL）、**有出处**（为什么）。

---

## 5. 与其它文档的分工（避免重复）

| 文档 | 管什么 | 与本文关系 |
|---|---|---|
| **`docs/CONVENTIONS.md`（本文）** | **规则本身**：什么能做、什么不能做、怎么自检 | 唯一来源 |
| `.clinerules` | 给 AI 的自动加载版（简短、命令化）| 必须与本文一致 |
| `docs/PROJECT_CONTEXT.md` | 项目背景、决策历史、§0.1 硬约束摘要 | 摘要 + 出处 |
| `docs/ENVIRONMENT_SETUP.md` | 环境怎么搭、坑怎么填（§2.6 = 约定 1 的细节）| 执行细节 |
| `docs/TEAM_ONBOARDING.md` | 新队友从零上手 | 面向新人 |
| `docs/INTERFACE_CONTRACT.md` / `OFFLINE_PIPELINE.md` / `RETARGETING_PIPELINE.md` | 接口与流水线用法、实测记录 | 业务细节 |

---

## 6. 版本

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-10-01 | 第 1 版 | 汇总历史要求（解释器 / 不改队友代码 / GBK / SSH / 不入库 / 基线复验）+ 新增「禁止往 C 盘写东西」；新增 `scripts/preflight.ps1` 与 `.clinerules` |



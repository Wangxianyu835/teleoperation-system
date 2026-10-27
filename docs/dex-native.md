# dex 原生手接入与阶段验收

范围：H1、GR1、G1 原生双手，Vector 优化，相机 world 与 Vision Pro 独立输入，PyBullet 仿真。
不连接实体机器人。DexPilot 留待下一轮。

## 实现索引

- `devtools/dex_native_env.ps1`：进程级 D 盘缓存与独立解释器路径。
- `devtools/bootstrap_dex_windows.py`：官方 wheel 的 Windows 依赖声明适配，不修改求解代码。
- `devtools/dex_native_acceptance.py`：阶段验收和原生手资产清单。
- `requirements-dex-native.lock.txt`：实验环境完整依赖锁定文件。
- `docs/dex-native-assets.json`：依赖版本、六手关节清单、资产及官方算法哈希。
- `docs/dex-native-runtime-lock.json`：Windows 编译运行时的官方 Conda 包 URL 和 SHA256。
- `robots/native_hand.py::NativeHandModel`：整机手部子树、固定指尖 frame、耦合限位与 FK 查询。
- `tests/test_dex_native.py`：本功能的契约及边界测试。

## 阶段记录

每阶段先保存 checkpoint，再运行对应验收并提交。结果位于 `outputs/tmp_dex_native/`。

| 阶段 | 修改前提交 | 检查点 | 状态 |
|---|---|---|---|
| 1 环境与基线 | fe9302e | checkpoint-20261027-1600-dex-stage1 | 通过 |
| 2 原生运动学 | 53c03c7 | checkpoint-20261027-1205-dex-stage2 | 通过 |

H1/G1 的固定指尖偏移取自本仓库 Inspire/Dex3 独立手 URDF；GR1 使用整机已有 tip frame。
H1 左手独立模型的食指 intermediate 原点与整机存在约 0.19 mm 差异：保留整机原点，仅移植末端固定偏移。
优化模型删除 visual/collision 资源引用，关节轴、原点及 mimic 保持整机定义；显示仍使用原整机模型。
阶段 2：六手各 100 个姿态（含耦合有效上下界）FK 位置误差均小于 0.1 mm，
全部查询恢复原整机位置和速度。新模型契约测试 5 项通过；原生回放和手部命令测试通过。

独立环境路径：`.venv/dex-native`。以现有正式环境的独立包副本为起点，再安装 dex 依赖；不修改正式环境。
原生手 URDF 的 mimic 从动关节不是独立优化变量：H1 为 6/12，GR1 为 6/11，G1 为 7/7。

Windows 的 PyPI 没有 `pin>=3.3.1` wheel。绑定由独立运行时 `.venv/dex-pin-bundle` 提供。
运行时从官方 Conda 包复制 Python 绑定及 DLL；不运行有注册副作用的环境安装事务。
本地构建标为 `dex-retargeting==0.5.0+winconda`：仅调整版本标记和 Windows 的 `pin` 依赖声明，
包内算法代码逐文件保持官方 wheel 字节不变。真实 Pinocchio 版本另外验证，不伪造 `pin` 包元数据。
wheel 来源、SHA256 与代码哈希记录在 `outputs/tmp_dex_native/wheels/*.provenance.json`。
正式项目环境不修改，缓存位于 D 盘。中断的 Conda 安装保留在被忽略目录中，不参与运行。

### 环境重建

先在仓库根目录加载 `devtools/dex_native_env.ps1`。创建 `.venv/dex-native`，
从正式环境复制独立 site-packages，或按完整 pip 锁定文件安装基础包。
本地 dex wheel 必须先按下列步骤构建，再使用完整锁定文件，不能从 PyPI 查找本地版本。

```powershell
. ./devtools/dex_native_env.ps1
# Anaconda 基础 Python 已包含 conda_package_handling，按 URL/SHA256 重建编译运行时。
& D:/Anaconda/python.exe -B devtools/bootstrap_dex_windows.py `
  --restore-lock docs/dex-native-runtime-lock.json --cache outputs/tmp_dex_native/cache/pip/lib/conda `
  --runtime .venv/dex-pin-bundle
& $dexPython -m pip download --no-deps dex-retargeting==0.5.0 -d outputs/tmp_dex_native/wheels
& $dexPython -B devtools/bootstrap_dex_windows.py `
  --wheel outputs/tmp_dex_native/wheels/dex_retargeting-0.5.0-py3-none-any.whl `
  --runtime .venv/dex-pin-bundle --environment .venv/dex-native
& $dexPython -m pip install --find-links outputs/tmp_dex_native/wheels -r requirements-dex-native.lock.txt
& $dexPython -B -m pip check
& $dexPython -B devtools/dex_native_acceptance.py --output outputs/tmp_dex_native/rebuilt-assets.json
```

原环境基线：448 项，0 失败、0 错误、1 原有跳过，146.519 秒。
独立实验环境基线：448 项，0 失败、0 错误、1 原有跳过，146.893 秒。
最终 Pinocchio 运行时接入后的完整基线再次通过：448 项，1 原有跳过，144.568 秒。
四条 CLI 冒烟退出码均为 0，pip check 无冲突，六手资产清单检查通过。
按 URL/SHA256 锁定文件在 `.venv/dex-pin-rebuilt` 重建编译运行时并独立导入验证成功。
官方 dex 0.5.0 发布源码提交为 `3f56141bc8bd2760d5e452e382937269554ebb21`，
安装后所有包内 Python 源码哈希与官方 wheel 逐一核对一致。
沙箱内首次测试遇到文件原子替换权限错误；允许该操作后原测试通过，不修改代码或测试来规避。
快速 preflight 仍报告用户原有 C 盘 pip/HuggingFace 缓存；本任务的七个进程缓存变量已重定向 D 盘，不删除用户缓存。

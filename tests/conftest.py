"""pytest 公共配置：第三方资产 / 可选依赖缺失时，跳过机械臂用例。

为什么放在这里（而不是去改队友的测试文件）
----------------------------------------
``tests/test_dual_arm.py`` 需要两样东西：

1. 第三方 TRON2A 描述包 ``third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf``
   （未随仓库分发，补齐方式见 ``docs/RETARGETING_PIPELINE.md`` 第 6.1 节）；
2. ``roboticstoolbox-python``（机械臂 IK 依赖）。

缺任何一样，该模块都会在收集或执行阶段直接 ERROR，把整个测试套件染红、掩盖其它真实失败。
本文件用 pytest 钩子在**不修改队友测试文件**的前提下把这些用例标记为 skip；
把上面两样东西补齐后，它们会自动恢复执行，无需再改任何代码。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# 与 tests/test_dual_arm.py 里的判据保持一致（相对路径，约定从仓库根目录运行 pytest）
TRON2A_URDF = Path("third_party/tron2-robot-description/tron2a/DACH_TRON2A/urdf/robot.urdf")
ARM_TEST_FILENAME = "test_dual_arm.py"


def _arm_prerequisite_reason() -> str | None:
    """返回「需要跳过机械臂用例」的原因；返回 None 表示先决条件齐备。"""
    if not TRON2A_URDF.is_file():
        return f"TRON2A URDF asset is missing: {TRON2A_URDF}"
    if importlib.util.find_spec("roboticstoolbox") is None:
        return "optional dependency roboticstoolbox-python is not installed"
    return None


def _is_arm_test_file(path: object) -> bool:
    return Path(str(path)).name == ARM_TEST_FILENAME


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """URDF 缺失时，把机械臂用例标记成 skip（模块本身能正常导入，所以能逐条标记）。"""
    reason = _arm_prerequisite_reason()
    if reason is None:
        return
    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        raw_path = getattr(item, "path", None) or item.fspath  # type: ignore[attr-defined]
        if _is_arm_test_file(raw_path):
            item.add_marker(skip_marker)


def pytest_ignore_collect(collection_path, config: pytest.Config) -> bool:
    """roboticstoolbox 未安装时，该模块连 import 都会失败，只能在收集阶段整文件忽略。"""
    if not _is_arm_test_file(collection_path):
        return False
    return importlib.util.find_spec("roboticstoolbox") is None


def pytest_terminal_summary(terminalreporter) -> None:
    """在汇总里说明机械臂用例为何没跑，避免被误读成「少测了」。"""
    reason = _arm_prerequisite_reason()
    if reason is None:
        return
    terminalreporter.write_sep("-", "conftest: arm tests not executed")
    terminalreporter.write_line(
        f"  tests/{ARM_TEST_FILENAME} skipped: {reason} (see docs/RETARGETING_PIPELINE.md section 6.1)"
    )

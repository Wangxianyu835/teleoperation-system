"""手部关键点缓冲区与「数据来源」常量的兼容入口（历史模块名）。

背景
----
``input_adapters/npy_replay_adapter.py`` 原先按 ``input_adapters.hand_keypoints``
导入 ``HandWindowBuffer`` / ``MEDIAPIPE_APPROX_SOURCE`` / ``NPY_REPLAY_SOURCE``，
但仓库里从来没有过这个模块（任何提交都搜不到），于是 ``NpyReplayAdapter``
一被导入就 ``ModuleNotFoundError`` —— 连带 ``scripts/validate_retarget_input.py``
也用不了。

处理方式（**不修改队友已入库的代码**）
------------------------------------
* 真正的实现统一在 ``retargeting/tracking.py``（正式模块，MediaPipe / VisionPro
  两个适配器都从那里取名字）。本文件只做「按旧路径转发」，不改动任何逻辑。
* ``NPY_REPLAY_SOURCE`` 在 ``retargeting/tracking.py`` 里没有（队友只定义了
  ``VISIONPRO_SOURCE`` / ``MEDIAPIPE_APPROX_SOURCE``）。为了不在队友文件上动刀，
  该常量在这里补上；它只是「数据来源」字符串标识，``tracking.py`` 与
  ``hand_core.py`` 都不做取值白名单校验，因此语义与放在 tracking 里一致。
  采用「先取队友的、取不到再兜底」的写法：队友哪天自己在 ``tracking.py``
  补上同名常量，这里会自动以他的为准，不会出现两处分叉。

用法
----
    from input_adapters.hand_keypoints import HandWindowBuffer
      等价于
    from retargeting.tracking import HandWindowBuffer
"""

from __future__ import annotations

from retargeting.tracking import (
    MEDIAPIPE_APPROX_SOURCE,
    VISIONPRO_SOURCE,
    HandIdentityTracker,
    HandWindowBuffer,
    ensure_hand25,
    mediapipe21_to_hand25,
    wrist_relative,
)

try:  # 队友若在 tracking.py 里补了该常量，自动以他的为准（保持单一事实来源）
    from retargeting.tracking import NPY_REPLAY_SOURCE
except ImportError:  # 当前 tracking.py 没有，兜底定义（见文件头说明）
    NPY_REPLAY_SOURCE = "npy_replay"

__all__ = [
    "MEDIAPIPE_APPROX_SOURCE",
    "NPY_REPLAY_SOURCE",
    "VISIONPRO_SOURCE",
    "HandIdentityTracker",
    "HandWindowBuffer",
    "ensure_hand25",
    "mediapipe21_to_hand25",
    "wrist_relative",
]

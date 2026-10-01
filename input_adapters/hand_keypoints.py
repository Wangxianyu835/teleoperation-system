"""手部关键点缓冲区与「数据来源」常量的兼容入口（历史模块名）。

背景
----
``input_adapters/npy_replay_adapter.py`` 原先按 ``input_adapters.hand_keypoints``
导入 ``HandWindowBuffer`` / ``MEDIAPIPE_APPROX_SOURCE`` / ``NPY_REPLAY_SOURCE``，
但仓库里从来没有过这个模块（任何提交都搜不到），于是 ``NpyReplayAdapter``
一被导入就 ``ModuleNotFoundError`` —— 连带 ``scripts/validate_retarget_input.py``
也用不了。

处理方式
--------
真正的实现统一在 ``retargeting/tracking.py``（正式模块，MediaPipe / VisionPro
两个适配器都从那里取名字）。本文件只做「按旧路径转发」，不改动任何逻辑：

    * 关键点转换、身份跟踪、三帧窗口 -> retargeting/tracking.py
    * 本文件额外保证 ``NPY_REPLAY_SOURCE`` 这个来源标识可用

用法
----
    from input_adapters.hand_keypoints import HandWindowBuffer
      等价于
    from retargeting.tracking import HandWindowBuffer
"""

from __future__ import annotations

from retargeting.tracking import (
    MEDIAPIPE_APPROX_SOURCE,
    NPY_REPLAY_SOURCE,
    VISIONPRO_SOURCE,
    HandIdentityTracker,
    HandWindowBuffer,
    ensure_hand25,
    mediapipe21_to_hand25,
    wrist_relative,
)

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

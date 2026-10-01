"""Input adapters：把【录制下来的】手部关键点日志喂给重定向链路。

现有成员
--------
    npy_replay_adapter.NpyReplayAdapter   .npy 日志回放（离线复现 / 单测用）
    hand_keypoints                        旧导入路径的兼容转发层（另含队友代码缺的 NPY_REPLAY_SOURCE 常量）

真正的关键点转换与三帧窗口实现在 retargeting/tracking.py（正式模块），
本包只负责「数据来源适配」。
"""

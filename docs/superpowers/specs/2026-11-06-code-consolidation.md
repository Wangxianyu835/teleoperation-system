# 当前代码清理与合并范围

状态：用户已批准执行，按配套计划分批实施。基线 `782712f3b4fb8e4c8b51268fe24ae33ebc7a3e4b`。

## 用户目标

完整核对当前代码，删除没有消费者的实现，合并重复且职责相同的内容，降低维护入口和重复实现的数量；先提供修改计划。

## 范围与证据

审查覆盖182个src Python文件、70个test_*.py测试模块、31个devtools目录条目，以及README、pyproject、现有Markdown和PowerShell入口。采用AST解析导入/相对导入、CLI注册字符串、符号读取、重复函数体、脚本入口和rg复查。AST的零引用结果必须人工排除动态注册、模块导出和外部直接入口；本轮没有运行全部设备工作流或重新执行完整测试。

没有确认可以直接整文件删除的孤立业务模块。`__main__.py`是Python入口；`applications/arm_validation.py`被README、测试和验收脚本使用；`applications/camera_record.py`被相机测试和脚本使用；`learning/hand_evaluation.py`被方法比较使用；`retargeting/hand/geometric.py`被verify_geometric_hand.py使用。它们都不能按“没有src导入”直接删除。

## 不变量

- 保留所有28个已注册CLI命令及其参数语义；允许修改内部handler路径。
- 保留公开输入、窗口和结果接口，包括无仓库调用但仍属于输入接口的MediaPipeCameraWorkflow.next_input。
- 相机与VR仍是并列来源，NN/dex独立选择；不改变坐标变换、三帧归一化、shared权重身份、耦合、滤波和IK数学。
- 保留整机命令、手结果、身体记录、TRON2A 48D、TransHandR诊断格式各自含义；不做自动旧数据转换。
- 单目标只构造所选目标；parts/side、独立holding、每组单次物理步进和机械臂GUI闪动修复必须通过回归。
- TransHandR上游数值和来源哈希不改，保留旧专用构造边界，不绕过权重检查。
- 不删除tests、assets、datasets、outputs、third_party或虚拟环境目录，不依据缺少测试判定无用。
- 合并只减少重复职责；不将不同单位、依赖或执行语义的实现合成万能模块。
- 执行时逐批遵守AGENTS检查点、最小回归/共享全套回归、提交和不推送政策；顺序执行，不创建子智能体。

## 交付

1. 删除15个无仓库消费者的函数/方法、1个无用常量和已复核的闲置导入/局部变量。
2. 迁移后删除5个薄文件，新增1个数组读取文件，净减少4个源码文件。
3. NN/dex实例构造一个维护位置，调用方继续负责来源、profile和checkpoint选项预检。
4. 提取预览/运行辅助，场景类不再携带大段UI实现；共享PyBullet初始化等待逻辑。
5. GBK扫描一个实现，保留脚本和CLI两种使用方式；共享绘图等比例坐标函数但保留原默认值。
6. Windows启动器共享准备逻辑，修正失效说明和检查项，不悄悄丢弃诊断参数。
7. 更新接口/入口地图、当前README和验收记录。历史计划保留并加后续变更说明。

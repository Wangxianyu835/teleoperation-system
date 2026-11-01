# 当前接口与旧接口清理

本次是明确的接口断开更新，不提供旧数据转换器。旧数据可从 Git checkpoint 查看。

修改前提交：`6e7b35d3c825f65edc16b5bfee99ca3080ac45a1`。
修改前 annotated tag：`checkpoint-20261101-1852-remove-legacy-demo-data`。

## 当前接口

- 观测：CameraFrame、RawHandFrame、UpperBodyObservation、RawTeleopObservation、WristObservation。
- 相机窗口：camera_palm_length_window3_v1，world米制→canonical right掌部→三帧→掌长中位数归一化。
- 算法结果：HandJointResult，目标手独立原生关节。
- 执行：HandJointCommand，经有效限位、滤波、限速、可选L21网格检查和耦合展开。
- 相机NN：每目标手型一份显式shared checkpoint，运行时适配左右侧。
- 手部结果：hand_joint_results_v1，分离raw与accepted command。回放校验规格、顺序和耦合。
- 身体记录：smplerx_body_recording_h5_v1，保存RGB、模型包及来源时间。
- TRON2A：身体观测和原生L21命令组合成48D，臂7+手17，每侧独立。
- VR：专用TransHandR预处理，内部原始18D，外部原生17D，诊断格式transhandr_vr_results_v2。

## 删除与迁移

删除全部applications/demo、旧根demo快捷入口、sim demo-joints和旧样例生成工具。
删除旧相机profile、旧模型加载器、无调用方的成对权重参数兼容层、hand realtime、replay l21-hand/native-hand/mounted-hand、历史18D角度文件读写。
删除L21到其他原装手的有损映射、旧结果writer/reader、glove分组读取和隐式时间轴。
删除明确列出的visual_hand_data、aligned_visual_hand_data、palm_local_visual_hand_data和旧demo样例H5。
当前canonical world、原始camera world和Vision Pro矩阵记录仍由当前入口使用，保留。
相机采集功能保留，camera_record委托当前world录制与对齐，不再生成18D角度副产物。
任务基准和物理仿真保留；geometric.py仅为L21几何诊断，不再登记为正式相机控制后端。
正式相机后端只有NN/dex。原始normalized坐标用于检测预览，不作为旧NN输入profile或执行文件。

三机器人NN使用重复的--hand-checkpoint MODEL PATH；dex不使用相机操作者标定。
TRON2A dual export使用--hand-joints读取新版结果，DualTeleoperator只接受RawTeleopObservation和显式control_seconds。

## VR 上游证据与边界适配

上游输入和原始网络输出保持不变。18→17转换仅在VR专用模型边界发生。
设备管理、源/目标手选择、命令处理和记录属于边界适配，不参与上游输入数值计算。
实际对照脚本devtools/verify_vr_both.py执行真实左分支与相邻注释右分支，各10个连续窗口、scale=1.0和1.7。
左右输入张量最大差异0，原始模型输出最大差异0。右分支不是上游真实右手硬件验收。

源文件SHA-256：

| 文件 | SHA-256 |
|---|---|
| `D:\2026\code\TransHandR\TransHandR\main_visionpro_realtime.py` | `f6141c0aa57b88e98a641912edaa3791c81116dd49a61119b8c4b811c200ff8f` |
| `D:\2026\code\TransHandR\TransHandR\model\model_poseformer.py` | `9c80ba61d4465e63c7444347dbb4bb7f94e86fb976b884dc2b24c3155d94e8a3` |
| `D:\2026\code\TransHandR\TransHandR\dataset\robot\l21_right\linkerhand_l21_right.urdf` | `7387e62b5f0e126b2a4915fff6c2230ecfcfd00e93f2f499e963095a082cbf01` |
| `D:\2026\code\TransHandR\TransHandR\dataset\data_collect\data_collect_visionpro.py` | `deb77448a16059a54e13d2a459c4d0805965614fefb263cf1235ffdbc80d8610` |
| `D:\2026\code\teleoperation-system-retargeting\assets\robots\l21\left\linkerhand_l21_left.urdf` | `cb265e5cad0e41bd02d9c60fe461c3b83645a7584cce175152328cc34bafc57a` |
| `D:\2026\code\teleoperation-system-retargeting\assets\robots\l21\right\linkerhand_l21_right.urdf` | `4c7d580b7518cb58277020ca9a1d4ff69dafce694a76617795be4fab64b66ba7` |

本地数值报告：outputs/tmp_legacy_cleanup/vr_upstream_equivalence.json。软件测试、数值等价与仿真探针不代替真实相机、头显或实体硬件验收。

## 本次验证

- teleoperation 环境运行 unittest discover -s tests：549项，547通过，2项因需要dex独立环境跳过。
- dex独立环境运行 test_normalized_dex 和 test_normalized_retargeters：5项全部通过，覆盖主环境跳过的2项。
- 三机器人场景原生命令探针：60个手部关节发生运动，非手部保持关节最大漂移0.0225rad，小于0.05rad阈值。
- VR实际上游源码与真实权重数值比较：左右输入与原始输出最大差异均0。
- 所有内部模块引用解析和 git diff --check 通过。

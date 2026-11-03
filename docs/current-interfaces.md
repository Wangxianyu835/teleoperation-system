# 当前接口与后续处理边界

## 正式手部控制

RGB与Vision Pro经各自设备适配器进入`hand_palm_length_window3_v1`，两者均可选择SharedJointNNRetargeter或NormalizedDexRetargeter，然后共用JointCommandPipeline、原生执行和hand_joint_results_v1。入口是`hand control --source camera|visionpro --backend nn|dex`；VR快捷入口默认NN，NN/dex均转发共同应用。

设备与方法分别选择。VR来源如实记录，不套用相机原始预处理，也不改写权重元数据。契约、命令和输入统一时的验收记录见[统一手部输入](unified-hand-input.md)。

## 当前接口

|边界|实际内容|
|---|---|
|观测|CameraFrame、RawHandFrame、UpperBodyObservation、RawTeleopObservation、WristObservation|
|共同手窗口|hand_palm_length_window3_v1；canonical right米制25点→三帧→掌长中位数归一化|
|相机数值规范|camera_palm_length_window3_v1；训练、三机器人相机、TRON2A和部分导出仍使用|
|NN权重|每目标手型一份显式shared checkpoint；右手canonical输出，运行时适配左右侧；严格校验规格、关节顺序、数值规范和有效限位|
|算法结果|HandJointResult，目标手独立原生关节|
|执行命令|HandJointCommand；有效限位、滤波、限速、可选L21网格检查和耦合展开|
|手部结果|hand_joint_results_v1；分离raw与accepted command，回放校验规格、顺序和耦合|
|身体记录|smplerx_body_recording_h5_v1；保存RGB、模型包及来源时间|
|TRON2A|身体观测和原生L21命令组合成48D，每侧7臂+17手|
|显式上游复现|hand vr --backend transhandr；内部原始18D、外部原生17D，诊断格式transhandr_vr_results_v2|

相机原始world、已对齐canonical world和Vision Pro矩阵记录仍由当前入口使用。相机采集保存观测并进行对齐；geometric.py仍供L21几何诊断使用。图像normalized坐标用于检测、预览及现存诊断，不等同于掌长归一化的无量纲后端窗口。

三机器人NN用重复的`--hand-checkpoint MODEL PATH`指定各目标shared权重，dex无需操作者标定。TRON2A导出用`--hand-joints`读取原生命令；DualTeleoperator接受RawTeleopObservation和显式control_seconds。正式手控制使用共同pipeline，但训练和双臂相机消费者尚未全部收敛到UnifiedHandPipeline。

## 保留并交由后续处理的部分

|剩余部分|当前消费者与边界|
|---|---|
|固定VR标定组件|hand dex-calibrate和devtools/dex_native_acceptance.py仍使用DexInputAdapter、DexCalibration、DexNativeRetargeter及相关辅助函数；正式NN/dex控制无需固定标定|
|相机pipeline与camera profile|MediaPipePalmLocalPipeline供三机器人和TRON2A预处理使用；camera profile还用于训练数据规范，不能直接删除|
|双臂legacy与normalized诊断分支|仍有旧行为测试或几何验收用途，本次未迁移|
|模型构造重复|hand_control._selection、hand_backend.build_camera_retargeter、dual.load_camera_hand_retargeter分别服务当前入口|
|GBK扫描工具|devtools与诊断CLI各有入口，尚未合并|
|录制与动作验证|活跃H5格式用途不同；PartialLimbVector仍用于拒绝不完整机器人动作|
|环境工具与路径接口|preflight.ps1仍需维护；training_run_directory当前侧参数行为保留|

本次只清理无消费者代码和混合文件中的闲置部分；以上活跃分支、上游TransHandR、FK、KalmanSmoother、身体跟踪和原生执行/回放继续保留。具体范围和执行记录见[清理计划](superpowers/plans/2026-11-04-unused-code-cleanup.md)。

## VR 上游证据与边界适配

上游输入和原始网络输出保持不变。18→17转换仅在VR专用模型边界发生。
设备管理、源/目标手选择、命令处理和记录属于边界适配，不参与上游输入数值计算。
输入统一改动时，devtools/verify_vr_both.py对真实左分支与相邻注释右分支各验证10个连续窗口、scale=1.0和1.7。本次无调用代码清理未重新运行该数值比较。
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

输入统一时的本地数值报告：outputs/unified_hand/vr_upstream_equivalence.json。软件测试、数值等价与仿真探针不代替真实相机、头显或实体硬件验收。

## 2026-11-04无调用代码清理验收

|验证项|本次实际结果|
|---|---|
|主环境完整unittest|557项，554通过、3项因缺少dex_retargeting跳过；94.143秒|
|独立dex环境回归|test_normalized_dex、test_normalized_retargeters、test_unified_hand_inputs共16项全部通过，无跳过；覆盖主环境跳过的3项|
|CLI入口|26个注册handler均能定位函数，全部--help命令返回0|
|源码检查|269个Python文件AST解析成功，退役对象无残余代码引用|
|删除范围|10个源码模块及2份已被替代的计划文档；混合文件中的闲置成员按范围删除|
|文档|8份Markdown本地链接检查无断链；当前文档不再提供退役接口用法|
|算法保持|保留函数/类AST对比通过，shared预处理、dex求解器和TransHandR数值路径未改动|

相较清理前560项，减少的3项仅验证已退出的旧最新帧包装器、旧VR控制循环和旧独立录制格式；输出维数、规格身份、限位篡改、左右手转换等校验已保留或迁移到当前shared接口。组合录制、统一VR缺手/恢复、通用结果和严格回放测试继续执行。

各批检查点、提交和剩余消费者见[执行记录](superpowers/plans/2026-11-04-unused-code-cleanup.md#执行记录)。没有启动真实相机、头显或实体机器人；本次未重新生成上游数值等价报告，历史证据在上节单独标注。

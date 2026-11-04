# 统一机器人输出与显式目标选择

日期：2026-11-04。源码基线：440f39e8322d28d94336b96609a7583ff2f47206。

## 已确认的目标

人体手部重定向和机械臂映射/IK分别生成目标机器人的关节命令，在统一输出层汇合，然后驱动H1-2、GR1-T2、G1或TRON2A＋L21。默认选择一台机器人；多机器人对比必须显式启用。当前只交付修改计划，不实施运行代码。

统一的是命令结构、调度与执行边界，不是四种机器人共用一份角度。目标选择发生在重定向/IK之前。每种机器人保留自己的HandSpec、权重、坐标映射、IK、限位、标定和状态。

## 当前缺口

- 手部已有HandJointResult→JointCommandPipeline→HandJointCommand，但直接调用场景写入。
- ArmSceneDriver.apply_each在一个方法里完成目标处理、IK和adapter.apply；相机及VR腕控制也直接调用该方法。
- TRON2A已汇合为Tron2AL21Command，但固定48D且拥有独立执行/录制流程。
- 三机器人模块未传--robot时加载所有机器人；这与新默认单目标要求不一致。
- ThreeRobotHandScene和Tron2ADualArmWorld各自创建连接、地面/物理配置与步进；不能简单叠放为四台对比。
- RobotLoader的数组顺序是左臂、右臂、左手、右手；TRON2A固定顺序是左臂、左手、右臂、右手，不能直接共用拼接代码。

## 范围和约束

- 默认单目标：新主入口必须显式给--robot；--robots只用于两个及以上不同目标，不指定目标时拒绝启动，不默认四台或暗选G1。
- 显示范围必须匹配目标和parts/side：单目标不出现其他机器人；both显示该目标整机，arms只显示对应机械臂，hands只显示对应手；left/right只显示所选侧的独立部位，both显示双侧；多目标只展示显式选中的目标。
- 目标ID固定为h1_2、gr1_t2、g1、tron2a_l21；目标自动绑定原装手或L21，常规机器人控制不再单独选择hand-model。
- 先保持现有重定向、归一化、权重校验、IK与安全数值；统一输出不再重复滤波或求解。
- 手、臂、左右侧独立状态；未更新部分保持上次接受命令或明确的初始化参考，不用零填充假装新观测。
- 控制时间使用统一会话相对秒与递增frame_id；来源时间保留原时基，异步身体结果不能冒充当前手帧。
- 同一控制周期所有目标命令先校验、再写入，每个物理连接只推进一次预定步数；不承诺运行时电机调用跨机器人事务回滚。
- 同一相机/SDK只采集一次，多目标共享观测；每目标独立权重、IK与命令状态。
- 四目标相机全臂全手均纳入交付；VR对H1-2/GR1-T2/G1沿用现有腕位姿控制，不伪造肩肘；TRON2A VR全臂当前无已验收映射，新入口启动前明确拒绝，新增算法不属于本次输出重构。
- 多机器人对比使用一个PyBullet连接、一个物理时钟和一个GUI；TRON2A与三机器人场景支持由外部注入连接和摆位，借用连接的对象不得disconnect公共连接。
- 保留现有手部输入/窗口/结果接口、shared权重和手H5；显式TransHandR路径与本地上游来源哈希保留，不改数值算法。
- 保留独立手部训练、导出和L21手场景；旧入口迁移到共同输出边界，不再复制正式控制循环，不在本次顺带清理其活跃legacy算法。
- 每批修改前检查Git状态并建立annotated checkpoint，测试后提交，不推送；共享行为改动执行完整unittest discover -s tests。

## 新主入口

```text
python -m teleoperation robot control --robot g1 --source camera --backend dex
python -m teleoperation robot control --robots h1_2 gr1_t2 g1 --source camera --backend dex
python -m teleoperation robot control --robots g1 tron2a_l21 --source camera --backend dex --tron2a-calibration <json>
python -m teleoperation robot control --robot g1 --source visionpro --ip <ip> --backend nn --hand-checkpoint g1 <shared.pth>
python -m teleoperation robot replay --input <robot_command_results_v1.h5> --robot g1
```

新增--parts hands|arms|both（默认both）、--side left|right|both（默认both）。NN按目标hand_model指定一份shared权重；arms-only不加载手模型。VR--parts both解释为腕位姿驱动机械臂加手指，报告标明wrist模式。camera身体模式需要SMPLer-X；hands-only不启动身体进程。TRON2A相机臂控制需要有效标定，不自动造identity标定。

## 单机器人及对应部位的显示

用户补充要求：只选一台机器人时，只出现该机器人或它对应的机械臂/手。不是只驱动一台、但仍把其他机器人摆在旁边。

|选择|可见对象|
|---|---|
|--robot g1 --parts both|一台完整G1|
|--robot g1 --parts arms --side right|只显示G1右机械臂|
|--robot g1 --parts hands --side right|只显示G1原装右手|
|--robot h1_2 --parts hands --side both|只显示H1-2原装左右手，不出现整机|
|--robot tron2a_l21 --parts arms|只显示TRON2A所选双臂，不加载L21可见手|
|--robot tron2a_l21 --parts hands|只显示所选L21手，不出现TRON2A整机|
|--robots g1 h1_2 --parts hands|只显示这两个目标所选侧的原装手|

parts作为默认显示范围，不新增一组必须填写的显示参数。both控制某一侧时整机仍按整机展示，未控制侧保持；arms/hands是独立部位展示。场景镜头围绕实际可见对象取景，单目标居中，不留三/四机器人布局空位。

独立部位使用原机器人URDF/HandSpec的几何和关节链；必要的运动学祖先可作为内部不可见、固定的支撑，不得显示其他身体部分或让未选部位的不可见碰撞体干扰控制。不得用移动相机裁剪冒充部位隔离，不把求解器简化代理模型当作原装手网格。部位场景的固定支撑/接触条件与整机不同，验收应核对FK、关节身份和控制命令，不声称动力学完全相同。

独立L21手模式不需要TRON2A身体标定或身体推理；这些依赖只在实际启用TRON2A机械臂时要求。录制保存parts/side及场景模式，回放恢复对应显示，不自动加载全身或全部目标。

## 验收与交付

先完成G1单目标闭环，再扩展H1-2、GR1-T2和TRON2A混合对比。离线合成观测、实际URDF、真实NN/dex/IK、PyBullet DIRECT验证命令与运动；真实设备验收单独标记，不将软件通过等同于相机/头显/硬件通过。

必须覆盖：拆分前后同初始状态的命令数值保持；求解不发电机/不步进；IK临时状态成功/失败均恢复；无双写与重复步进；非控制关节保持；缺手不阻断有效机械臂；单目标和多目标隔离；TRON2A安装/姿态/标定不变；新命令H5严格校验、无覆盖、回放不重求解/滤波。

显示验收同时检查各body所属目标、可见link集合以及真实渲染图；不能用“只有一台在运动”或getNumBodies()==1代替。双手和TRON2A附加手可能包含多个body，但目标身份只有一个。


## 录制与回放边界

`robot_command_results_v1` 只记录已经通过目标规格校验并提交给输出层的接受命令。回放不会重新调用手部 NN/dex、SMPLer-X、腕姿态求解、IK 或滤波器；它只恢复记录的目标部位显示、命令帧和物理步数。

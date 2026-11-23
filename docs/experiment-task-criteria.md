# 当前 30 项操作实验：流程与精确成功判据

核对日期：2026-11-23。源码基线：`e0eb35ff41e287782bef19512cda7c469d45e44c`；修改前检查点为 `checkpoint-20261123-1151-experiment-walkthrough`。本文只解释当前任务实现与直接影响判据的场景覆盖，不改任务代码，也不把任务意图写成已经实现的验证条件。

当前注册分组为 **L1 5 项、L2 11 项、L3 9 项、L4 5 项，共 30 项**，以 [registry.py:39](../src/teleoperation/simulation/tasks/registry.py#L39) 为准。每个任务实现 `_load_objects()`、`_set_initial_positions()` 和 `check_success()`；`reset()` 依次清场、加载、摆放并清除 `success` 标志，见 [base.py:29](../src/teleoperation/simulation/tasks/base.py#L29)。表中的操作流程解释实验意图，布尔表达式才是程序实际检查的条件。

本次执行 `task_eval --verify --robot all` 得到退出码 0、`90/90 tasks evaluable, 0 excluded`，本机证据保存在 [verify_walkthrough_20261123.json](../outputs/task_eval/verify_walkthrough_20261123.json)（运行产物不随文档提交，可按主说明重跑生成）。这验证的是场景稳定、无复位假成功、人工构造目标状态后判据可触发、物体在可达范围内；`verify_task()` 会直接调用 oracle 改物体位姿，因此 **90/90 不是机器人完成 30 项操作的成功率**，见 [task_eval.py:3224](../devtools/task_eval.py#L3224)。本次没有采集真人操作实验。

## 读表约定：位置、角度与“在上方”

- `p_o=(x_o,y_o,z_o)` 是物体 `o` 的当前 base 世界坐标；`r_o,pitch_o,yaw_o` 是由其当前四元数转成的欧拉角，单位为弧度。位置单位均为米。源码读取 base 位姿，见 [base.py:98](../src/teleoperation/simulation/tasks/base.py#L98)。
- `D(a,b)=norm(p_a-p_b)` 是三维欧氏距离；`dxy(a,b)=norm(p_a[:2]-p_b[:2])` 只看世界坐标 XY 平面。前者实现见 [base.py:112](../src/teleoperation/simulation/tasks/base.py#L112)。
- `A(a,b,t)` 精确代表 `_is_above(a,b,threshold=t)`：`dxy(a,b)<t and z_a-z_b>0`，见 [base.py:118](../src/teleoperation/simulation/tasks/base.py#L118)。这里的 `threshold` **只限制水平偏差**；Z 只要求中心更高，没有高度上限，不检查物体底面、容器开口、接触或支撑。故把东西悬在容器很高处也可能满足“放入/放到”的原始判据。
- `T` 表示**当前**桌子的 AABB 顶面高度 `getAABB(table)[1][2]`。默认桌子中心 Z 为 `0.4`、半厚 `0.02`，原始顶面约 `0.42`；场景层会调整实际位置，见 [base.py:250](../src/teleoperation/simulation/tasks/base.py#L250)。
- 下表的 `and` 表示各条件同时成立，所有 `<`、`>` 都是严格不等式；恰好等于阈值不满足对应条件。`pos_threshold` 默认 `0.05`；虽然基类还有 `angle_threshold=5.0` 度，这 30 个任务的成功函数没有使用统一的 5 度阈值，见 [base.py:21](../src/teleoperation/simulation/tasks/base.py#L21) 及各行判据。

## L1：基础操作，5 项

| 任务 ID | 实验意图与操作过程 | 当前 `check_success()` 的精确条件 | 判据没有检查的动作或状态 |
|---|---|---|---|
| `pushcube` | 接近方块，沿桌面推到蓝色目标；原始方块 `(0.5,-0.2,0.43)`，目标 `(0.3,0.3,0.43)`。 | `D(cube,target)<self.pos_threshold`，默认 `<0.05`。[basic.py:22](../src/teleoperation/simulation/tasks/basic.py#L22) | 没有限制只能推，抓起搬过去也可；没有要求整个方块进入区域、释放或停止。 |
| `pickcube` | 闭指抓住方块，抬高；当前方块边长 `0.04`、质量 `0.04 kg`。名称注释虽写“拿起并放下”，检测发生在举高时。 | 存在 `table` 且 `z_cube>T+0.18`；没有桌子则为假。[basic.py:33](../src/teleoperation/simulation/tasks/basic.py#L33)、[basic.py:44](../src/teleoperation/simulation/tasks/basic.py#L44) | 不检查抓握接触、先拿起后放下或松手；也不是从方块初始中心提升 `0.18`。 |
| `pickplacecube` | 抓住方块，移到盘子上方，下降并松手。 | `A(cube,plate,0.05)`。[basic.py:76](../src/teleoperation/simulation/tasks/basic.py#L76) | 不检查先抬起、落在盘面、稳定支撑或松手。 |
| `uprearcup` | 杯子初始绕 Y 轴侧放 `pi/2`；抓取并扶正，竖直放置。 | `abs(r_cup)<0.2 and abs(pitch_cup)<0.2 and z_cup>0.42`；角度界限约 `11.46°`。[basic.py:89](../src/teleoperation/simulation/tasks/basic.py#L89)、[basic.py:96](../src/teleoperation/simulation/tasks/basic.py#L96) | 不检查 yaw、桌面接触、放置位置或释放；Z 是世界坐标阈值。 |
| `balltrashcan` | 拿球，移到桶口，下降并释放进桶。 | `dxy(ball,trashcan)<0.04 and z_ball<z_trashcan+0.08`。[basic.py:122](../src/teleoperation/simulation/tasks/basic.py#L122) | 没有 Z 下界、入桶轨迹、松手或桶内体积判定；原始桶和 camera 桶的几何不同，见下文。 |

## L2：工具与单物体交互，11 项

| 任务 ID | 实验意图与操作过程 | 当前 `check_success()` 的精确条件 | 判据没有检查的动作或状态 |
|---|---|---|---|
| `rotatefaucet` | 抓住龙头手柄，绕竖直轴转动约四分之一圈。 | `abs(yaw_handle)>pi/2-0.1`，即 `>1.4708 rad`、约 `84.27°`。[tools.py:26](../src/teleoperation/simulation/tasks/tools.py#L26) | 不要求恰好 `90°`，也不检查手柄仍位于基座、接触或水流；检测当前世界 yaw，不累计旋转量。 |
| `rotatehearth` | 接触炉灶旋钮并转动；有机器人时要维持机器人与旋钮接触。 | `(robot_id is None or bool(getContactPoints(robot,knob))) and abs(yaw_knob)>0.5`；`0.5 rad≈28.65°`。[tools.py:45](../src/teleoperation/simulation/tasks/tools.py#L45) | 没有“向下按”的位移/力阈值，也不要求特定手指接触；无机器人审计跳过接触检查。 |
| `openmicrowave` | 将原始关闭门向外拉；原始门与炉体中心 X 差 `0.16`。 | `abs(x_door-x_microwave_body)>0.20`；若沿原始正 X 方向拉，则相对原始位置需多于 `0.04`。[tools.py:71](../src/teleoperation/simulation/tasks/tools.py#L71)、[tools.py:79](../src/teleoperation/simulation/tasks/tools.py#L79) | 不检查铰链角度、Y/Z 对齐或沿指定方向运动；反侧足够远也满足。 |
| `closemicrowave` | 门初始拉开（X 差 `0.30`）；将其推回。 | `abs(x_door-x_microwave_body)<0.02`。[tools.py:97](../src/teleoperation/simulation/tasks/tools.py#L97)、[tools.py:106](../src/teleoperation/simulation/tasks/tools.py#L106) | 检查中心 X 对齐，不是与实体前表面贴合；没有 Y/Z、姿态或锁止条件。 |
| `opendrawer` | 拉动薄抽屉面板；原始关闭 X 差 `0.165`。 | `abs(x_drawer-x_cabinet)>0.19`；若向原始正 X 拉，需额外多于 `0.025`。[tools.py:124](../src/teleoperation/simulation/tasks/tools.py#L124)、[tools.py:132](../src/teleoperation/simulation/tasks/tools.py#L132) | 不检查导轨、Y/Z、打开方向或保持在柜上；没有保存初始 X 再相减。 |
| `closedrawer` | 抽屉初始 X 差 `0.30`；向柜体推回。 | `abs(x_drawer-x_cabinet)<0.17`。[tools.py:150](../src/teleoperation/simulation/tasks/tools.py#L150)、[tools.py:159](../src/teleoperation/simulation/tasks/tools.py#L159) | 不检查 Y/Z、姿态或真实容纳关系；评估场景将原始厚抽屉换成薄面板。 |
| `liftmug` | 抓起**杯盖**并离开杯体；原始盖心比杯心高 `0.055`。 | `abs(z_lid-z_mug_body)>0.075`；杯体不动且向上抬时，需额外多于 `0.020`。[tools.py:175](../src/teleoperation/simulation/tasks/tools.py#L175)、[tools.py:183](../src/teleoperation/simulation/tasks/tools.py#L183) | 名字不表示抬整个杯子；绝对值也允许盖比杯心低超过阈值，不检查 XY、抓住盖或释放。 |
| `openlaptop` | 抬起笔记本屏幕，转成打开姿态。 | `abs(r_laptop_screen)>pi/3`，即 `>60°`。[tools.py:210](../src/teleoperation/simulation/tasks/tools.py#L210) | 不检查屏幕与底座的相对铰链角、连接关系或位置；只看屏幕世界 roll。 |
| `ballmug` | 拿球移到杯口，下降放入。 | `dxy(ball,mug)<0.03 and z_ball>z_mug+0.04`。[tools.py:234](../src/teleoperation/simulation/tasks/tools.py#L234) | 只有高度下界，没有上界；悬在很高处也满足，不检查进入杯腔或松手。 |
| `pourwater` | 抬起水壶，向杯子倾倒，再扶正放回。 | `abs(r_kettle)>pi/3 and z_kettle>0.5`，即世界 roll 超过 `60°` 且壶心世界 Z 超过 `0.5`。[tools.py:259](../src/teleoperation/simulation/tasks/tools.py#L259) | 不读取杯子位置，不检查壶嘴朝向、液体、体积、倾倒流向、扶正或放回；pitch 方向的倾斜也不等同于所测 roll。 |
| `breadtoaster` | 将面包移到烤面包机上，向机身内推动前面按钮。 | `A(bread,toaster,0.04) and y_button<y_toaster+0.095`；原始按钮 Y 差 `0.11`，若机身不动，向内位移需 `>0.015`。[tools.py:281](../src/teleoperation/simulation/tasks/tools.py#L281)、[tools.py:292](../src/teleoperation/simulation/tasks/tools.py#L292) | 不检查面包进入槽内、按钮 Z/接触、先放面包后按键或烘烤；源码“至少 3 cm”注释与表达式的 `1.5 cm` 差值不同。 |

## L3：双手与多物体协作，9 项

| 任务 ID | 实验意图与操作过程 | 当前 `check_success()` 的精确条件 | 判据没有检查的动作或状态 |
|---|---|---|---|
| `ballbimanual` | 一手拿球，送到身体中间，由另一手接球。 | `abs(y_ball)<0.1 and z_ball>0.6`；两个都是世界坐标限制。[bimanual.py:18](../src/teleoperation/simulation/tasks/bimanual.py#L18) | 不检查 X、左右手接触、先后持球、交接或接收方释放；单手举球也可能满足。 |
| `potbimanual` | 双手抓锅柄并抬锅。 | 存在 `table` 且 `z_pot>T+0.15`；无桌子为假。[bimanual.py:49](../src/teleoperation/simulation/tasks/bimanual.py#L49) | 不检查双手、锅柄、锅姿态或放回；高度是锅心距当前桌面，不是比初始锅心提升 `0.15`。 |
| `pottomato` | 按意图一手抬锅，另一手把番茄放进去。 | `A(tomato,pot,0.04)`。[bimanual.py:79](../src/teleoperation/simulation/tasks/bimanual.py#L79) | 不检查锅被抬起、两手协作、入锅深度或松手。 |
| `pottray` | 双手搬锅，移到托盘上，放下。 | `A(pot,tray,0.06)`。[bimanual.py:101](../src/teleoperation/simulation/tasks/bimanual.py#L101) | 不检查双手、托盘接触、锅水平、释放或稳定承重。 |
| `stackboxes` | 搬起一个箱子，放到另一个上方。 | `dxy(box_a,box_b)<0.04 and 0.04<abs(z_box_a-z_box_b)<0.07`。[bimanual.py:123](../src/teleoperation/simulation/tasks/bimanual.py#L123) | 不指定谁在上，不检查姿态、接触或松手。两个箱子边长均 `0.10`，直立面对面叠放中心差约 `0.10`，与此高度窗口不同，不能把触发判据等同于标准叠箱。 |
| `panhearth` | 移开锅盖，将平底锅放到炉灶。 | `A(pan,hearth,0.05) and dxy(pan,lid)>0.05`。[bimanual.py:155](../src/teleoperation/simulation/tasks/bimanual.py#L155) | 不检查先开盖后搬锅或接触；单独把锅搬走、盖留原地，也可能满足盖已移开的末态条件。 |
| `tidyuptable` | 依次把球形 `item1`、方形 `item2` 收到篮子。 | `A(item1,basket,0.06) and A(item2,basket,0.06)`。[bimanual.py:187](../src/teleoperation/simulation/tasks/bimanual.py#L187) | 不检查放入顺序、真实篮内体积、释放或两手分工。 |
| `pottomatoout` | 揭锅盖，从锅内取出番茄，放到桌面。 | `dxy(tomato,pot)>0.06 and z_tomato>z_pot-0.06`。[bimanual.py:220](../src/teleoperation/simulation/tasks/bimanual.py#L220) | 不读取盖子状态，也不检查桌面接触或释放；水平阈值 `0.06` 小于锅外半径 `0.08`，不是完整几何出锅判定。 |
| `plateoven` | 打开烤箱门，放入盘子，再关门。 | `abs(x_plate-x_oven)<0.1 and abs(y_plate-y_oven)<0.1 and abs(x_oven_door-x_oven)<0.22`。[bimanual.py:251](../src/teleoperation/simulation/tasks/bimanual.py#L251) | 没有盘子 Z/姿态或门 Y/Z 条件，不验证先开后关；原始门 X 差 `0.20`，其“关门”子条件初始已满足，整个任务还须盘子 XY 到位。 |

## L4：顺序与精细组合操作，5 项

| 任务 ID | 实验意图与操作过程 | 当前 `check_success()` 的精确条件 | 判据没有检查的动作或状态 |
|---|---|---|---|
| `pottomatoplate` | 开锅盖，取番茄，移到盘子并松手。 | `A(tomato,plate,0.04)`。[sequences.py:33](../src/teleoperation/simulation/tasks/sequences.py#L33) | 不读取锅或盖子的状态，不检查开盖、取出、动作顺序、盘面接触或释放。 |
| `penbrushpot` | 抓笔并移到笔筒口，插入再松手。 | `A(pen,pot_holder,0.03)`。[sequences.py:55](../src/teleoperation/simulation/tasks/sequences.py#L55) | 不检查笔的朝向、插入深度、笔尖位置、接触或释放。 |
| `drawerbook` | 拉开抽屉，放入书，再推回抽屉。 | `A(book,drawer,0.06) and abs(x_drawer-x_cabinet)<0.17`。[sequences.py:82](../src/teleoperation/simulation/tasks/sequences.py#L82) | 不检查抽屉曾打开、书在抽屉内部或松手；原始 X 差 `0.14`，评估覆盖后起点差 `0.155`，“关上”子条件可在开始就为真。 |
| `twistbottlecaps` | 拿住瓶子，将盖对准瓶口并旋紧。 | `A(cap,bottle,0.02)`。[sequences.py:111](../src/teleoperation/simulation/tasks/sequences.py#L111) | 不检查瓶子被拿起、盖的角度、旋转过程、螺纹啮合或紧固；仅盖心在瓶心上方且水平靠近。 |
| `stacktoyblocks` | 按顺序拿起并叠放三块积木。 | `dxy(block_a,block_b)+dxy(block_b,block_c)<0.08 and max(z_a,z_b,z_c)-min(z_a,z_b,z_c)>0.06`。[sequences.py:138](../src/teleoperation/simulation/tasks/sequences.py#L138) | 不指定 A/B/C 的上下次序，不要求三种高度、相邻接触、竖直姿态、释放或按顺序操作；其中两块同高也可能满足。 |

## 初始状态、相对位置与世界阈值不能混用

`pickcube` 的“相对桌面 +0.18”与 `potbimanual` 的“相对桌面 +0.15”引用的是当前桌面 AABB；搬高工作台后阈值会一起升高。它们没有计算 `z_now-z_initial`。默认直立方块中心在台面上约 `0.02`，锅心约 `0.03`，因此从自然支撑位起的中心提升量分别约须超过 `0.16` 与 `0.12`；这些是几何推算，不是额外判据。原始 `_set_initial_positions()` 的 Z 可能尚未消除碰撞嵌入，实际 trial 应查看 `initial_objects`。[basic.py:33](../src/teleoperation/simulation/tasks/basic.py#L33)、[bimanual.py:30](../src/teleoperation/simulation/tasks/bimanual.py#L30)、[task_eval.py:3122](../devtools/task_eval.py#L3122)

门、抽屉、盖、按钮的表达式都用**当前两个物体的坐标差**；上表从原始摆位算出的“额外 2 cm / 2.5 cm / 4 cm / 1.5 cm”只在参照物不动、沿指定方向移动时成立。场景调整或参照物被推动会改变这层解释。源码没有为这些成功函数保存初始位姿并计算行程；当前角度条件也比较世界欧拉角，而非相对初始角或相对基座角。[tools.py:17](../src/teleoperation/simulation/tasks/tools.py#L17)、[tools.py:175](../src/teleoperation/simulation/tasks/tools.py#L175)、[tools.py:281](../src/teleoperation/simulation/tasks/tools.py#L281)

`uprearcup` 仍保留世界 `z>0.42`，`pourwater` 保留世界 `z>0.5`，`ballbimanual` 保留世界 `z>0.6` 及 `abs(y)<0.1`；这些不会因为桌子移动自动变成“离桌高度”。相反，`A`、两物体中心差与 `D` 在两物体一起平移时保持不变；场景层压缩间距、分离碰撞或单独摆放后则未必保持不变。[basic.py:96](../src/teleoperation/simulation/tasks/basic.py#L96)、[tools.py:259](../src/teleoperation/simulation/tasks/tools.py#L259)、[bimanual.py:18](../src/teleoperation/simulation/tasks/bimanual.py#L18)、[task_eval.py:1373](../devtools/task_eval.py#L1373)

## 代理几何与评估场景覆盖

任务普遍使用盒子、实心圆柱和球等代理物体。`load_cylinder()` 直接创建 `GEOM_CYLINDER`，不会因名称叫“杯/锅/篮子”自动生成内腔，见 [base.py:150](../src/teleoperation/simulation/tasks/base.py#L150)。例如原始 `balltrashcan` 注释虽说空心，实际调用实心圆柱；`ballmug`、`pottomato`、`tidyuptable`、`penbrushpot` 也不能仅凭名称视为真实容器，见对应任务行及加载代码。

`pottomatoout` 已明确改用 `load_hollow_cylinder()`：固定锅半径 `0.08`、全高 `0.08`、壁厚/底厚各 `0.008`，默认八段桶壁；原始番茄放在锅内。评估器把 `('pot','tomato')` 登记为容器关系，跳过通常的 AABB 重叠分离，见 [bimanual.py:198](../src/teleoperation/simulation/tasks/bimanual.py#L198)、[base.py:174](../src/teleoperation/simulation/tasks/base.py#L174)、[task_eval.py:125](../devtools/task_eval.py#L125)、[task_eval.py:581](../devtools/task_eval.py#L581)。这项修复没有把其他任务的实心锅一并改为空心锅。

`SCENE_OPS` 的当前覆盖如下；`place` 中 `None` 表示保留该坐标。覆盖先于整体平移，后续还会落台、分离和压缩布局，因此这里不是最终 `initial_objects`。[task_eval.py:104](../devtools/task_eval.py#L104)、[task_eval.py:445](../devtools/task_eval.py#L445)、[task_eval.py:1373](../devtools/task_eval.py#L1373)

| 任务 | 几何和原始坐标覆盖 |
|---|---|
| `closedrawer` | 抽屉半尺寸改为 `(0.015,0.18,0.1)`，放到 `(0.62,0.3,None)`，覆盖时相对原始柜心 X 差 `0.22`。 |
| `drawerbook` | 同样的薄面板，放到 `(0.555,0.3,None)`，X 差 `0.155`，已在 `<0.17` 的关闭窗口内。 |
| `opendrawer` | 同样的薄面板，放到 `(0.565,0.3,None)`，X 差 `0.165`；任务源码本身也已采用此薄面板和 `>0.19` 阈值。 |

此外 camera 场景对 `balltrashcan` 会把桶替换为固定的八壁空心容器（质量 `0`），区别于原始任务的可动实心圆柱；这是 camera 专用摆法，不能推广到所有运行模式，见 [task_eval.py:1191](../devtools/task_eval.py#L1191)、[task_eval.py:1220](../devtools/task_eval.py#L1220)、[task_eval.py:3096](../devtools/task_eval.py#L3096)。它仍调用同一个位置判据，并未自动加入“已松手”或完整体积判定。

## 旧截图中的排除名单与当前代码

若截图显示“24/30 可评测、以下 6 项排除”，它对应仓库中保留的旧记录，见 [2026-11-18 记录:63](session-2026-11-18-camera-task-validation.md#L63)。当前可执行定义为 `CRITERIA_INVALID={}`、`CRITERIA_ABSOLUTE_Z={}`、`SCENE_UNSUPPORTED={}`，且 `SCENE_MODE={}`，见 [task_eval.py:98](../devtools/task_eval.py#L98)、[task_eval.py:116](../devtools/task_eval.py#L116)。空字典表示当前没有预先排除项，不表示所有判据已完整表达任务自然语言含义。

| 旧记录的问题 | 当前源码事实 |
|---|---|
| `liftmug`、`opendrawer`、`openmicrowave` 复位即成功 | 盖高度差阈值改 `>0.075`；薄抽屉关位差 `0.165`、开阈值 `>0.19`；微波炉关位差 `0.16`、开阈值 `>0.20`。具体严格条件见上表。 |
| `rotatehearth` 要把旋钮按进桌面 | 已改成机器人接触（有机器人时）与 `abs(yaw)>0.5`；没有旧世界 Z 下压条件。 |
| `breadtoaster` 要把按钮按到桌面以下 | 已改为相对机身的 Y 条件 `<y_toaster+0.095`；没有旧世界 Z 条件。实际原始行程差为 `>1.5 cm`，不是注释写的 `3 cm`。 |
| `pottomatoout` 实心锅容不下番茄 | 已使用空心固定锅，场景层也保留锅与番茄的容纳关系。 |
| `pickcube` 必须世界 `z>0.66`；`potbimanual` 必须世界 `z>0.58` 或降机器人 | 当前分别为 `z_cube>T+0.18` 和 `z_pot>T+0.15`；`SCENE_MODE` 为空。截图和遗留注释中的绝对值不能用来解释当前成功函数。 |

`task_eval.py` 文件开头仍保留“72/90”和“3 invalid + 2 absolute + 1 unsupported”的历史说明，其 `SCENE_OPS` 注释也仍提到旧 `opendrawer>0.12`，`shift_oracle()` 文档字符串仍提到旧 `pickcube z>0.66`；这些与当前执行代码不同，见 [task_eval.py:25](../devtools/task_eval.py#L25)、[task_eval.py:105](../devtools/task_eval.py#L105)、[task_eval.py:1448](../devtools/task_eval.py#L1448)。判断当前行为应看上表各 `return`、当前字典与本轮验证报告。

## 如何解释成功结果的边界

上述 30 个原始成功函数都不要求操作者松手；命名为“双手”的任务没有检查左右手各自接触或交接历史；命名为“顺序”的任务没有阶段状态机，只检查当前末态。尤其不能据 `pourwater` 断言液体已倒入杯中、据 `twistbottlecaps` 断言瓶盖旋紧、据 `A(...)` 断言物体已被容器接住。这些结论来自上表所列完整成功函数。

评估器另外要求默认连续 `120` 个物理步成立（`DT=1/240`，即 `0.5 s`），并拒绝任一当前质量大于零的任务物体中心低于 `T-0.08` 的状态；场景准备后若判据已经成立，trial 最终强制记失败。见 [task_eval.py:85](../devtools/task_eval.py#L85)、[task_eval.py:2777](../devtools/task_eval.py#L2777)、[task_eval.py:2824](../devtools/task_eval.py#L2824)、[task_eval.py:3150](../devtools/task_eval.py#L3150)。这些附加规则没有增加松手、动作顺序或真实双手检查；“未低于桌面”也不等于正在与桌子接触。运行方式、策略与辅助机制应另随实验报告说明。

# 操作者关联与失效状态修复

本次修复针对 `arm_diagnose.h5` 中的故障：原始采集样本 547 将一位操作者检出两个框，
分数 0.997409 和 0.534732，原策略立即永久锁定，后续单框也不恢复。
此外桥的 trace 读取上次有效控制状态，持续显示旧数值及空 reason。

## 新的选择与恢复规则

前置检测器仍按分数 >= 0.5、宽 >= 50、高 >= 150 像素过滤。改变发生在过滤后的操作者选择边界：

1. 没有既定操作者时，只接受唯一框。初次出现多个框时不按大小或分数挑选。
2. 已有操作者、多框情况下，将每个框与上次接受的操作者框比较 IoU。
   最佳匹配必须 IoU >= 0.5，且比第二名至少高 0.25，才沿用该框。
   这允许本次局部假框存在时继续跟随已关联的人；明确关联成立时，可以有其他检测框。
3. 无法明确关联时输出 `operator ambiguous`，停止产生有效身体结果，机械臂保持。
4. 短暂歧义消失后，恢复候选必须与旧操作者 IoU >= 0.5，连续三帧彼此 IoU >= 0.5
   才恢复；缺框、再次歧义或不一致会打断恢复计数。显示 `operator recovering: n/3`。
5. 超过两秒没有已接受身份，或启动歧义持续超过两秒，仍永久锁定本会话。
   单框突然换到与原框 IoU < 0.1 的位置，仍拒绝并锁定。重新运行并重新标定解除锁定。

这里使用空间连续性，不是人脸、服装或生物身份识别。两个人交叉、遮挡、同位置替换时
仍有识别能力限制；有竞争匹配时宁可保持，不承诺复杂多人跟踪。
本次没有提高检测阈值，也没有修改上游 S32 网络、权重、RGB 处理、裁剪或 SMPL-X 数值推理。
上游来源与哈希仍见 `docs/smplerx-arm.md`。

身份恢复通过后，身体跟踪器原有的三帧稳定恢复仍生效。因此实际机械臂恢复可能还要等待
额外身体样本；身份 n/3 与身体恢复是两项检查，不能将等待帧当作新身体样本计数。
原有 0.5 秒身体过期、关节/工作空间/速率限制保持不变。

工作进程结果 metadata 增加 `operator_selection`，记录 accepted_count、selected_index、
policy、recovering、recovery_count、locked；旧键和接口保持兼容。
探针区分 `operator_ambiguous`、`operator_recovering` 和 `operator_rejected`。

## 当前状态显示

桥为每次 update 保存单独的当前显示快照，继续保留最后有效数值用于保持。
trace 保留旧字段并新增 held、tracking_state、fresh、age_s、sample_id。
HUD 显示 `S/tracking`、`S/hold` 或 `S/lost_hold`；S 只表示来源。
控制台 arm trace 显示状态、保持标志、是否新样本、年龄、序号及真实 reason。
预览也显示当前跟踪状态和年龄。reset 会清空显示快照。

跟踪器在有效样本过期时继续输出 lost_hold，同时保留当前模型的拒绝原因；
等待下一结果期间不会把 `operator ambiguous` 等原因替换成通用的“没有新结果”。
恢复后显示 fresh=True、held=False；cal=ok 仍仅表示中性标定已完成。

## 验证

回归覆盖：真实故障框、初始多人禁止自动选择、短暂歧义三帧恢复、恢复窗口打断、
空间无关的人不能接管、持续歧义/长时间缺席锁定、过期状态保留原始原因、
当前 HUD 与 trace 显示保持且保持的目标位置不变。

真实 S32 验证采用 GPU 预热后的独立会话：重复原始样本 545 五次初始化体型，
随后推理原始样本 547 和 549。三帧均返回双侧 valid=True；547 的 metadata 为
accepted_count=2、policy=previous_operator_iou、locked=False。
报告为 `outputs/tmp_realtime/operator_recovery_real_model.json`。
这验证原始故障画面的处理，不替代动态现场、真实多人及交叉遮挡验收。
全量 `python -m unittest discover -s tests`：576 项，跳过 3 项，无失败。
相同 dex 环境下真实 worker 8 帧往返通过，H5 往返最大误差为 0，报告位于
`outputs/tmp_realtime/operator_recovery_roundtrip/report.json`。

退出旧程序再启动原命令即可生效，不需要重新下载模型。旧 H5 保存的是当时观测，
直接回放旧 H5 不会重新执行新检测器选择；需要用原始 RGB 重新推理才能验证身份选择。

"""把若干份 realtime_teleop --report JSON 的关键数字打成长表（A/B 用，不碰 pybullet）。

用法::

    E:\\python3.11.7\\python.exe outputs\\tmp_realtime\\report_table.py ab_all_on.json ab_recovery_off.json

每个文件一段，每个机器人每侧一行：
    req/opp   前臂方向误差中位（度）/ 反向帧占比
    up_err    上臂方向误差中位（度）
    etask     肘任务残差（mm, median/max）—— 求解器有没有兑现肘目标
    jerk      上臂/前臂逐帧方向变化（度, p90/max）—— 「抽」的量
    cmdjerk   命令自己逐帧变化（度, max）—— 输入侧
    rec/rej   支路恢复救回/拒绝次数
"""

import json
import sys

SIDES = (("h1_2", "left"), ("h1_2", "right"), ("gr1_t2", "left"),
         ("gr1_t2", "right"), ("g1", "left"), ("g1", "right"))


def numbers(summary):
    """一行里的几个数（缺字段给 None，报告里看得出「没量到」）。"""
    def pick(key, stat):
        value = (summary or {}).get(key)
        return None if not isinstance(value, dict) else value.get(stat)
    return {
        "up_err": pick("upper_dir_error_deg", "median"),
        "fore_err": pick("forearm_dir_error_deg", "median"),
        "opp": summary.get("forearm_opposite_fraction") if summary else None,
        "etask": pick("elbow_task_mm", "median"),
        "etask_max": pick("elbow_task_mm", "max"),
        "wtask": pick("wrist_task_mm", "median"),
        "ujerk": pick("upper_jerk_deg", "max"),
        "fjerk": pick("forearm_jerk_deg", "max"),
        "ejerk": pick("elbow_jerk_mm", "max"),
        "cjerk": pick("command_jerk_deg", "max"),
        "rec": summary.get("elbow_recoveries") if summary else None,
        "rej": summary.get("branch_rejections") if summary else None,
    }


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        posture = ((report.get("arm") or {}).get("posture") or {})
        counts = ((report.get("bridge") or {}).get("counts") or {})
        print("== {}".format(path))
        print("   bridge counts: " + json.dumps(counts, ensure_ascii=False))
        print("   robot/side            up_err fore_err  opp | etask(med/max) wtask |"
              " jerk up/fore/elbow (max) cmdjerk | rec/rej")
        for robot, side in SIDES:
            line = numbers(posture.get(robot, {}).get(side))
            print("   {:8s} {:5s} {:>7} {:>8} {:>5} | {:>7} {:>7} {:>5} |"
                  " {:>6} {:>6} {:>6} {:>7} | {:>3} {:>3}".format(
                      robot, side, line["up_err"], line["fore_err"], line["opp"],
                      line["etask"], line["etask_max"], line["wtask"],
                      line["ujerk"], line["fjerk"], line["ejerk"], line["cjerk"],
                      line["rec"], line["rej"]))
        print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

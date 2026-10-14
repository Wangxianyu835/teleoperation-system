"""把若干份 realtime_teleop --report JSON 的关键数字打成长表（A/B 用，不碰 pybullet）。

用法::

    E:\\python3.11.7\\python.exe devtools\\report_table.py ab_chain.json ab_posture.json

每个文件一段，每个机器人每侧一行：
    up_err/fore_err  上臂/前臂方向误差中位（度）—— 「跟前臂弯曲方向对不对」看 fore_err
    opp   前臂反向帧占比（> 0 就是「弯反了」）
    sh    肩关节行程（度）—— 「大臂抬不抬得起来」看它
    upt   上臂方向行程（度）—— 同一个问题的方向口径（与关节角无关）
    span  肘跨距（mm）—— 「张不张得开怀抱」看它
    etask/wtask 肘/腕任务残差（mm, median/max）—— 求解器有没有兑现目标
    clamp 这一侧被逐轴行程夹断的帧数（bridge.counts.clamped）
    jerk  上臂/前臂/肘逐帧变化（度、mm，max）—— 「抽」的量；cmdjerk 是命令自己
    rec/rej       支路恢复救回/拒绝次数
    缺字段打 "-"（chain 口径不给肘任务，这些列本来就是空的，不是 0）
"""

import json
import sys

SIDES = (("h1_2", "left"), ("h1_2", "right"), ("gr1_t2", "left"),
         ("gr1_t2", "right"), ("g1", "left"), ("g1", "right"))


def numbers(summary, counts, side):
    """一行里的几个数（缺字段给 None，表里打成 "-"，不跟 0 混）。"""
    def pick(key, stat, source=None):
        value = (summary if source is None else source or {}).get(key)
        return None if not isinstance(value, dict) else value.get(stat)
    clamped = ((counts or {}).get("clamped") or {})
    return {
        "up_err": pick("upper_dir_error_deg", "median"),
        "fore_err": pick("forearm_dir_error_deg", "median"),
        "opp": summary.get("forearm_opposite_fraction") if summary else None,
        "shoulder": summary.get("shoulder_travel_deg") if summary else None,
        "upper": summary.get("upper_arm_dir_travel_deg") if summary else None,
        "span": summary.get("elbow_span_mm") if summary else None,
        "etask": pick("elbow_task_mm", "median"),
        "etask_max": pick("elbow_task_mm", "max"),
        "wtask": pick("wrist_task_mm", "median"),
        "ujerk": pick("upper_jerk_deg", "max"),
        "fjerk": pick("forearm_jerk_deg", "max"),
        "ejerk": pick("elbow_jerk_mm", "max"),
        "cjerk": pick("command_jerk_deg", "max"),
        "rec": summary.get("elbow_recoveries") if summary else None,
        "rej": summary.get("branch_rejections") if summary else None,
        "clamp": clamped.get(side),
    }


def _fmt(value, spec=">7"):
    """None -> "-"：chain 口径的肘任务列没有数字，不能打成 0。"""
    if value is None:
        return "{:>{width}}".format("-", width=spec.lstrip(">"))
    return format(value, spec)


def main(paths):
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        posture = ((report.get("arm") or {}).get("posture") or {})
        bridge = (report.get("bridge") or {})
        counts = (bridge.get("counts") or {})
        print("== {}".format(path))
        print("   bridge: map_mode={} calibrate_mode={} travel_mode={}".format(
            bridge.get("map_mode"), bridge.get("calibrate_mode"),
            (report.get("arm") or {}).get("arm_travel_mode")))
        print("   bridge counts: " + json.dumps(counts, ensure_ascii=False))
        print("   robot/side   up_err fore_err    opp | sh(deg) up(deg) span(mm) |"
              " etask(med/max)  wtask | clamp | jerk up/fore/elbow  cmdjerk | rec/rej")
        for robot, side in SIDES:
            line = numbers(posture.get(robot, {}).get(side), counts, side)
            print("   {:8s} {:5s} {:>9} {:>8} {:>6} | {:>7} {:>7} {:>8} |"
                  " {:>7} {:>7} {:>6} | {:>5} | {:>7} {:>7} {:>7} {:>7} | {:>3} {:>3}"
                  .format(robot, side, _fmt(line["up_err"], ">9"),
                          _fmt(line["fore_err"], ">8"),
                          _fmt(line["opp"], ">6"),
                          _fmt(line["shoulder"]), _fmt(line["upper"]),
                          _fmt(line["span"], ">8"),
                          _fmt(line["etask"]), _fmt(line["etask_max"]),
                          _fmt(line["wtask"], ">6"),
                          _fmt(line["clamp"], ">5"),
                          _fmt(line["ujerk"]), _fmt(line["fjerk"]),
                          _fmt(line["ejerk"]), _fmt(line["cjerk"], ">7"),
                          _fmt(line["rec"], ">3"), _fmt(line["rej"], ">3")))
        print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])

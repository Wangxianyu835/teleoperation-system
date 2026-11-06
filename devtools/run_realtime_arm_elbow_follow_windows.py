"""SMPLer-X arm preset with explicit operator/off elbow reference selection."""
import os
import sys
import windows_runtime as runtime


FLAG = "--arm-elbow-follow"


THIS_BUILD = "operator"        # 默认启用操作者肘弯参考


OFF_SWITCH = "--off"           # 本文件的糖，等价 --arm-elbow-follow off


def apply_pin(argv: list[str]) -> list[str]:
    """把这一版的口径钉进参数表；``--off`` 与显式 ``--arm-elbow-follow`` 优先。"""
    argv = list(argv)
    wanted = "off" if OFF_SWITCH in argv else THIS_BUILD
    argv = [item for item in argv if item != OFF_SWITCH]
    if any(item == FLAG or item.startswith(FLAG + "=") for item in argv):
        return argv                     # 你已经自己写了这一项，不插手
    return argv + [FLAG, wanted]


def pinned_value(argv: list[str]) -> str:
    """参数表里最终生效的那一档（argparse 的规矩：后面的覆盖前面的）。"""
    value = THIS_BUILD
    for index, item in enumerate(argv):
        if item == FLAG and index + 1 < len(argv):
            value = argv[index + 1]
        elif item.startswith(FLAG + "="):
            value = item.split("=", 1)[1]
    return value



def main(argv):
    argv = apply_pin(argv)
    runtime.log('INFO', f'{FLAG} {pinned_value(argv)}')
    return runtime.run_camera_launcher(argv, with_arms=True, script_path=__file__)


if __name__ == '__main__':
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)

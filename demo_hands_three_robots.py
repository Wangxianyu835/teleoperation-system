"""三台机器人手部演示 —— PyCharm 里右键本文件 -> Run，直接出画面（无需参数）

运行效果
--------
    H1-2 / GR1-T2 / G1 三台论文机器人并排站立，
    各自用【原装】灵巧手，按同一份采集数据同步屈伸。
      手指：由数据驱动（L21 18 维 -> 各机器人原装手降维映射）
      手臂：数据里没有手臂关节角，保持中性姿态【静止】（当前算法范围）

这份文件是什么
--------------
    真正干活的是 scripts/show_hands_all.py，但它默认是【无头模式】（不弹窗），
    必须手动加 --render 才会出画面 —— 所以在 PyCharm 里右键运行它会"看不到东西"。
    本文件就是给 PyCharm 准备的一键入口：
      * 默认打开 GUI（--render）+ 无限循环（--loop 0）
      * 默认自动挑 datasets/raw 下已采好的那份数据
      * 在 PyCharm 的 Run 面板里不用配任何东西
      * 解释器若缺 pybullet/numpy/h5py，自动兜底用仓库内 lib/

用法
----
    PyCharm ：右键本文件 -> Run 'demo_hands_three_robots'
    命令行  ：python demo_hands_three_robots.py
              python demo_hands_three_robots.py --view front     # 正面视角
              python demo_hands_three_robots.py --speed 0.5      # 半速
              python demo_hands_three_robots.py --file datasets/raw/xxx.h5
    参数与 scripts/show_hands_all.py 完全一致：本文件把参数【原样透传】，
    命令里写的参数覆盖内置默认值（如 --view front 覆盖默认的 full）。

退出
----
    关掉 PyBullet 窗口；或在 PyCharm 的 Run 面板点红色方块（停止）。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(ROOT, 'scripts')
# 回放程序在 scripts/ 下（不是包，按模块导入），仓库根目录用于 envs/ teleop/
for _p in (SCRIPTS, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 默认数据（按优先级取第一份存在的）：自己采的 809 帧优先，其次队友重定向数据
CANDIDATES = [
    os.path.join(ROOT, 'datasets', 'raw', 'my_recording_angles.h5'),
    os.path.join(ROOT, 'datasets', 'raw', 'retarget_twohand_153542.h5'),
]

# 运行本演示需要的依赖
DEPS = ('pybullet', 'numpy', 'h5py')

# 内置默认参数（写在 --file 之前；用户命令行参数追加在后面，后写的生效）
DEFAULTS = ['--render', '--loop', '0']


def _missing(names):
    """返回 names 里当前解释器导入不到的模块名"""
    import importlib.util
    return [n for n in names if importlib.util.find_spec(n) is None]


def _ensure_deps():
    """确保 pybullet/numpy/h5py 可用；缺了就兜底，兜不住就给出可操作的提示"""
    missing = _missing(DEPS)
    if not missing:
        return
    lib = os.path.join(ROOT, 'lib')
    if os.path.isdir(lib):
        # lib/ 是 "pip install -r requirements.txt -t lib" 的产物（不入库）
        sys.path.insert(0, lib)
        if not _missing(DEPS):
            print('[提示] 当前解释器缺 %s，已自动改用仓库内 lib/'
                  % '/'.join(missing))
            print('       （想彻底解决：PyCharm > Settings > Project > '
                  'Python Interpreter 换成 E:\\python3.11.7\\python.exe）')
            return
    raise SystemExit(
        '缺少依赖：%s\n'
        '  任选一种方式：\n'
        '    1) PyCharm：Settings > Project: simulation_platform > '
        'Python Interpreter\n'
        '       选装好 pybullet 的那个（开发机是 '
        'E:\\python3.11.7\\python.exe，不是仓库里的 .venv）\n'
        '    2) 命令行：E:\\python3.11.7\\python.exe -m pip install -r '
        'requirements.txt\n'
        % '/'.join(missing))


def _pick_file():
    """自动挑一份已采好的数据（和 show_hands_all.py 的 --file 同格式）"""
    for path in CANDIDATES:
        if os.path.exists(path):
            return path
    raise SystemExit(
        '没找到可用的数据文件，请用 --file 指定，例如：\n'
        '  python demo_hands_three_robots.py --file datasets/raw/xxx.h5\n'
        '已尝试：\n  ' + '\n  '.join(CANDIDATES))


def _has_file_arg(argv):
    return any(a == '--file' or a.startswith('--file=') for a in argv)


def main():
    # PyCharm 的 Run 控制台（以及重定向到文件时）不是 tty，Python 默认会
    # 整块缓冲 -> 画面都出来了、控制台还是一片空白。改成按行刷新。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding='utf-8', line_buffering=True)
        except Exception:
            pass

    _ensure_deps()

    user = list(sys.argv[1:])
    forward = list(DEFAULTS)
    if not _has_file_arg(user):
        forward += ['--file', _pick_file()]
    # 参数原样透传给 show_hands_all：argparse 对重复选项取【最后一个】，
    # 所以写在后面的用户参数会覆盖前面的内置默认值
    sys.argv = [sys.argv[0]] + forward + user

    print('=' * 72)
    print('一键入口 demo_hands_three_robots.py -> scripts/show_hands_all.py')
    print('=' * 72)
    print('  参数：%s' % ' '.join(sys.argv[1:]))
    print('  画面：三台机器人并排，原装手按同一份数据同步屈伸')
    print('  退出：关掉 PyBullet 窗口，或点 PyCharm Run 面板的红色方块')
    print()

    import show_hands_all
    show_hands_all.main()


if __name__ == '__main__':
    main()

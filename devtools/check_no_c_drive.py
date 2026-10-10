"""检查「C 盘零写入」硬约束是否还在生效

背景
----
用户明确要求：**项目相关的东西一律不许落在 C 盘**
（代码、依赖、pip/HF/matplotlib/torch 缓存、临时文件都不行）。

历史上真实翻车过一次：一次 `pip install -r requirements-retargeting.txt`
把 **216.78 MB** 下载缓存写进了 `C:\\Users\\<用户>\\AppData\\Local\\pip\\Cache`；
`robot_descriptions`（RTB 取 URDF 用）还会往 `C:\\Users\\<用户>\\.cache\\huggingface\\hub`
下机器人模型。根因是这些工具默认缓存目录都在 C 盘，而环境变量没重定向。

修法 = 把缓存/临时目录全部指到 E 盘（用户级环境变量），见：
    devtools/env_e_drive_cache.ps1
    docs/CONVENTIONS.md 约定 1（硬约束，current-state 布局的唯一来源）
    （旧布局的 ENVIRONMENT.md / PROJECT_CONTEXT.md 已随上游删除，不再作为依据）

用法（在项目根目录执行）
------------------------
    python devtools/check_no_c_drive.py            # 只报告
    python devtools/check_no_c_drive.py --strict   # 有 FAIL 就退出码 1

判定
----
[OK]   环境变量已指向非 C 盘，且该默认 C 盘目录不存在或为空
[FAIL] 环境变量未设置（工具会自己回落到 C 盘）或仍指向 C 盘
[FAIL] C 盘上仍残留**非空**的默认缓存目录（先搬走或清掉）
[WARN] C 盘残留目录为空（无害，但建议删掉以免被再次写满）
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 需要重定向的环境变量 -> 说明
ENV_VARS = [
    ('PIP_CACHE_DIR', 'pip 下载缓存'),
    ('MPLCONFIGDIR', 'matplotlib 字体/配置缓存'),
    ('HF_HOME', 'HuggingFace(hub/xet) 缓存，robot_descriptions 下模型会用它'),
    ('TORCH_HOME', 'torch.hub 权重缓存'),
    ('XDG_CACHE_HOME', '通用 XDG 缓存'),
    ('TEMP', '临时文件'),
    ('TMP', '临时文件'),
]


def _rel(path):
    home = os.path.expanduser('~')
    if path.lower().startswith(home.lower()):
        return '~' + path[len(home):]
    return path


def _nonempty(path):
    try:
        return any(True for _ in os.scandir(path))
    except Exception:
        return False


def _dir_size(path):
    total = 0
    for base, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(base, fn))
            except Exception:
                pass
    return total


def check_env():
    bad = []
    print('── 环境变量（必须已设置且不落在 C:）──')
    for name, note in ENV_VARS:
        val = os.environ.get(name, '')
        if not val:
            print(f'  [FAIL] {name:<16} 未设置 -> 会回落到 C 盘默认值（{note}）')
            bad.append(name)
        elif val[:2].upper() == 'C:':
            print(f'  [FAIL] {name:<16} = {val}  <- 就在 C 盘（{note}）')
            bad.append(name)
        else:
            print(f'  [OK  ] {name:<16} = {val}')
    return bad


def check_leftovers():
    home = os.path.expanduser('~')
    cands = [
        os.path.join(home, 'AppData', 'Local', 'pip', 'Cache'),
        os.path.join(home, 'AppData', 'Local', 'pip'),
        os.path.join(home, 'AppData', 'Local', 'matplotlib'),
        os.path.join(home, '.cache', 'huggingface'),
        os.path.join(home, '.cache', 'pip'),
        os.path.join(home, '.cache', 'torch'),
        os.path.join(home, '.torch'),
    ]
    bad, warn = [], []
    print('── C 盘默认缓存位置残留检查 ──')
    for p in cands:
        if not os.path.isdir(p):
            print(f'  [OK  ] 不存在            {_rel(p)}')
            continue
        if _nonempty(p):
            mb = _dir_size(p) / 1024 / 1024
            print(f'  [FAIL] 仍有内容 {mb:8.2f} MB  {_rel(p)}')
            bad.append(p)
        else:
            print(f'  [WARN] 空目录            {_rel(p)}')
            warn.append(p)
    return bad, warn


def check_project_location():
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    ok = root[:2].upper() != 'C:'
    mark = '[OK  ]' if ok else '[FAIL]'
    print('── 项目位置 ──')
    print(f'  {mark} 项目根目录 {root}')
    return [] if ok else [root]


def main():
    ap = argparse.ArgumentParser(description='检查项目相关产物是否违反「C 盘零写入」硬约束')
    ap.add_argument('--strict', action='store_true', help='有 FAIL 时退出码 1')
    args = ap.parse_args()

    print('=' * 62)
    print('C 盘零写入自检（详见 docs/ENVIRONMENT_SETUP.md 第 2.6 节）')
    print('=' * 62)

    bad = []
    bad += check_project_location()
    bad += check_env()
    left_bad, left_warn = check_leftovers()
    bad += left_bad

    print('-' * 62)
    if not bad:
        print(f'结论：[OK] 未发现 C 盘写入隐患（{len(left_warn)} 个空目录可顺手删）')
        return 0
    print(f'结论：[FAIL] {len(bad)} 处隐患，修法：')
    print('  PowerShell:  . .\\devtools\\env_e_drive_cache.ps1 -Persist')
    print('  搬运缓存:    robocopy "<C 盘旧目录>" E:\\cache\\<同名> /E /MOVE /R:1 /W:1')
    return 1 if args.strict else 0


if __name__ == '__main__':
    sys.exit(main())

"""提交前检查：源码里有没有「GBK 无法编码」的字符

背景
----
中文 Windows 控制台默认 **GBK (cp936)**。若代码里 print() 了
非 GBK 字符（如 U+2713 对勾、U+26A0 警告、U+2192 箭头、U+00B2/B3 上标等），
Python 会直接抛：

    UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'

**导致程序崩溃**，而不是仅仅显示乱码。

用法（在项目根目录执行）
------------------------
    python -m teleoperation tools check-gbk-safe            # 只报告
    python -m teleoperation tools check-gbk-safe --strict   # 有违规就退出码 1（可用于 CI）

推荐的 ASCII 替代写法（见下方 SUGGEST 映射表）
----------------------------------------------
    U+2713 -> [OK]      U+2717 -> [FAIL]     U+26A0 -> 注意
    U+2192 -> ->        U+2190 -> <-         U+2194 -> <->
    U+2265 -> >=        U+2264 -> <=         U+2248 -> ~=
    U+00B2 -> ^2        U+00B3 -> ^3         U+00B9 -> ^1
    U+25C0 -> <--       U+25B6 -> -->        U+FE0F -> 删除
"""
import argparse
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

SKIP_DIRS = {'lib', '.venv', '__pycache__', 'linkerhand_sdk', 'robots', '.git'}

SUGGEST = {
    '\u2713': '[OK]', '\u2717': '[FAIL]', '\u26a0': '注意',
    '\u2192': '->', '\u2190': '<-', '\u2194': '<->',
    '\u2265': '>=', '\u2264': '<=', '\u2248': '~=',
    '\u00b2': '^2', '\u00b3': '^3', '\u00b9': '^1',
    '\u25c0': '<--', '\u25b6': '-->', '\ufe0f': '(删除)',
}


def scan(root='.'):
    bad = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(base, fn)
            try:
                with open(path, encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f'  READ_FAIL {path}: {e}')
                continue
            for lineno, line in enumerate(lines, 1):
                for ch in set(line):
                    try:
                        ch.encode('gbk')
                    except UnicodeEncodeError:
                        bad.append((path, lineno, ch, line.strip()[:70]))
    return bad


def main(args=None):
    if args is None:
        raise TypeError("Use teleoperation.cli.main() to parse command arguments")

    print('=' * 90)
    print('GBK 安全性检查（中文 Windows 控制台为 cp936）')
    print('=' * 90)

    bad = scan(args.root)
    if not bad:
        print('  未发现 GBK 无法编码的字符  [OK]')
        print('=' * 90)
        return 0

    print(f'  发现 {len(bad)} 处问题：')
    print()
    seen = set()
    for path, lineno, ch, snippet in bad:
        key = (path, ch)
        if key in seen:
            continue
        seen.add(key)
        sug = SUGGEST.get(ch, '(建议改用 ASCII)')
        print(f'  {path}:{lineno}  U+{ord(ch):04X} {ch!r}  ->  建议: {sug}')
        print(f'      {snippet}')
    print()
    print(f'  共 {len(bad)} 处，分布在 {len({p for p, _, _, _ in bad})} 个文件')
    print('=' * 90)

    if args.strict:
        print('  [FAIL] --strict 模式：存在违规，退出码 1')
        return 1
    print('  提示：这些字符在中文 Windows 控制台 print() 时会抛 '
          'UnicodeEncodeError 导致崩溃')
    return 0



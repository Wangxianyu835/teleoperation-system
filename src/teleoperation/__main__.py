# 模块定义；链路：python -m teleoperation→CLI入口
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

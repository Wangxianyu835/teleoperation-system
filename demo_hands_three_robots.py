"""Launch the maintained three-robot native-hand viewer."""
from teleoperation.cli import main

if __name__ == "__main__":
    import sys
    raise SystemExit(main(["tools", "show-all-hands", *sys.argv[1:]]))

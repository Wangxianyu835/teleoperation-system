"""SMPLer-X camera arm diagnostic launcher; select --robot explicitly.

Shared setup and --check live in windows_runtime. The public robot control
command is recommended for normal use; this entry retains preview diagnostics.
"""
import os
import sys
import windows_runtime as runtime


def main(argv):
    return runtime.run_camera_launcher(argv, with_arms=True, script_path=__file__)


if __name__ == '__main__':
    status = main(sys.argv[1:])
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)

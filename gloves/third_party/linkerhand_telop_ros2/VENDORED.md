# Vendored Linker FFG transport

The FFG serial parser and protocol constants under
`src/ffg_capture/transport/vendor/` are copied from:

- Repository: `https://github.com/linker-bot/linkerhand-telop-ros2`
- SDK version: `2.12.10`
- Source commit: `7c2f1b097dedf3c12a80270bf3b910f9c31396e8`
- Source files:
  - `src/linkerhand_retarget/linkerhand_retarget/linkerhand/linkerforce.py`
  - `src/linkerhand_retarget/linkerhand_retarget/linkerhand/constants.py`

Only the serial transport, FFG frame parser, version query, and force-response
reading are used. ROS2 nodes, robot-hand retargeting, motor commands, and MuJoCo
display code are not vendored.

The upstream repository and files are licensed under the MIT License. See
`LICENSE` in this directory.

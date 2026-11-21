"""One command policy for independent coordinates from any hand algorithm."""
import math
import time

import numpy as np

from teleoperation.contracts.hand_control import HandJointResult, HandJointCommand
from .postprocessing import KalmanSmoother


class JointCommandPipeline:
    """把独立关节求解结果变成可执行命令；同一对象跨帧复用。

    previous 保存上次接受角度，filtered 保存滤波角度，times 与
    _last_command_times 区分有效输入时间和已发命令时间（包括缺手保持）。
    因此相同输入角度在不同历史/时间间隔下可能产生不同命令。
    """
    def __init__(self, specs, filter_mode="lowpass", alpha=.2, max_velocity=1.5,
                 kalman_q=1., kalman_r=.01, initial=None, guards=None, clock=time.monotonic):
        if filter_mode not in ("lowpass", "kalman", "none"):
            raise ValueError("Unknown joint filter")
        if not np.isfinite([alpha, max_velocity, kalman_q, kalman_r]).all() or not 0 < alpha <= 1 or min(max_velocity, kalman_q, kalman_r) <= 0:
            raise ValueError("Invalid joint command parameters")
        self.specs, self.mode, self.alpha = dict(specs), filter_mode, float(alpha)
        if any(key != spec.key for key, spec in self.specs.items()):
            raise ValueError("Command specification keys differ from target identities")
        self.clock, self.guards = clock, guards or {}
        self.q, self.r = float(kalman_q), float(kalman_r)
        self.previous, self.filtered, self.times, self.filters, self.stats = {}, {}, {}, {}, {}
        self._last_command_times = {key:None for key in specs}
        self.speeds = {key: np.minimum(spec.velocity_limits, max_velocity) for key, spec in specs.items()}
        for key, spec in self.specs.items():
            values = np.array((initial or {}).get(key, spec.q_reference), dtype=float, copy=True)
            spec.expand(values)
            self.previous[key] = values
            self.times[key] = None
            self.stats[key] = {"commands": 0, "held_frames": 0, "rate_limited": 0, "limit_clipped": 0}
            self._seed(key)

    def _seed(self, key):
        self.filtered[key] = self.previous[key].copy()
        self.filters[key] = KalmanSmoother(process_noise=self.q, measurement_noise=self.r, gate=0.,
                                          max_velocity=float(np.max(self.speeds[key])))
        self.filters[key].step(self.previous[key])

    def _command(self, key, timestamp):
        spec = self.specs[key]
        named = spec.expand(self.previous[key])
        return HandJointCommand(*key, spec.spec_hash, spec.joint_names,
                                np.array([named[n] for n in spec.joint_names]), float(timestamp))

    def missing(self, key, timestamp=None):
        """缺失时保持上一角度，并重置滤波状态；不把关节送回零位。"""
        now = self.clock() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("Command timestamp must be finite seconds")
        last = self._last_command_times[key]
        if last is not None and now <= last:
            return self._command(key,now)
        self._last_command_times[key] = now
        self.stats[key]["held_frames"] += 1
        self.times[key] = None
        self._seed(key)
        return self._command(key, now)

    def update(self, result: HandJointResult, timestamp=None) -> HandJointCommand:
        if not isinstance(result, HandJointResult) or result.key not in self.specs:
            raise ValueError("Command pipeline requires a result for an active hand")
        key, spec = result.key, self.specs[result.key]
        if result.spec_hash != spec.spec_hash or result.joint_names != spec.independent_joint_names:
            raise ValueError("Result does not match the target joint specification")
        now = self.clock() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            raise ValueError("Command timestamp must be finite seconds")
        if not result.valid:
            return self.missing(key, now)
        previous_time = self.times[key]
        last_command = self._last_command_times[key]
        if last_command is not None and now <= last_command:
            return self._command(key, now)
        elapsed = 1/30 if previous_time is None else now - previous_time
        if previous_time is None and last_command is not None:
            # Holding frames are still emitted commands. A 26 ms reacquisition
            # must not use a nominal 33 ms step and exceed the URDF speed.
            elapsed = min(elapsed,now-last_command)
        if elapsed > .25:
            self._seed(key)
        dt = min(elapsed, .1)
        self.times[key] = now
        self._last_command_times[key] = now
        # bounds 为 (D,2)；转置并用 * 解包成各关节下限/上限，clip 逐项截断。
        target = np.clip(result.positions, *spec.bounds.T)
        self.stats[key]["limit_clipped"] += int(not np.allclose(target, result.positions))
        if self.mode == "lowpass":
            self.filtered[key] += (1 - (1-self.alpha)**(dt*30)) * (target-self.filtered[key])
            filtered = self.filtered[key]
        elif self.mode == "kalman":
            self.filters[key].dt = dt
            filtered = self.filters[key].step(target)
        else:
            filtered = target
        filtered = np.clip(filtered, *spec.bounds.T)
        before = self.previous[key]
        delta = filtered-before
        active = np.abs(delta) > 1e-14
        # 所有关节共享一个缩放比例，既满足各自速度上限，又保留本次运动方向。
        fraction = min(1., float(np.min(self.speeds[key][active]*dt/np.abs(delta[active])))) if active.any() else 1.
        command = before + fraction*delta
        guard = self.guards.get(key)
        if guard is not None:
            # Mesh queries use the same independent native coordinates.
            projected = np.asarray(guard.project(before, command))
            if not np.allclose(projected, command, atol=1e-9):
                self.previous[key] = projected.copy()
                self._seed(key)
            command = projected
        self.previous[key] = command.copy()
        self.stats[key]["commands"] += 1
        self.stats[key]["rate_limited"] += int(fraction < 1.)
        return self._command(key, now)

    def summary(self):
        return {"filter": self.mode, "alpha_at_30hz": self.alpha,
                "joint_space": "independent native coordinates",
                "max_velocity_rad_s": {"/".join(key): self.speeds[key].tolist() for key in self.specs},
                "sides": {"/".join(key): dict(value) for key, value in self.stats.items()},
                "collision_guard": {"/".join(key): guard.summary() for key, guard in self.guards.items()}}

# 函数 checkpoint_options/checkpoint_metadata/checkpoint_description（校验）；链路：CLI/调用方→应用编排→输入/算法/存储/仿真
"""Translate workflow arguments into model selection and provenance."""

from teleoperation.contracts.checkpoints import validate_checkpoint_selection


def checkpoint_options(args, *, prefix="", default=None):
    stem = f"{prefix}_" if prefix else ""
    paths = [getattr(args, stem + name, None) for name in
             ("checkpoint", "left_checkpoint", "right_checkpoint")]
    if all(path is None for path in paths):
        paths[0] = default
    validate_checkpoint_selection(*paths, required=True)
    return dict(zip(("checkpoint_path", "left_checkpoint", "right_checkpoint"),
                    (None if path is None else str(path) for path in paths)))


def checkpoint_metadata(options):
    if options["checkpoint_path"] is not None:
        return {"checkpoint": options["checkpoint_path"]}
    return {name: options[name] for name in ("left_checkpoint", "right_checkpoint")}


def checkpoint_description(options):
    return " ".join(f"{name}={path}" for name, path in checkpoint_metadata(options).items())

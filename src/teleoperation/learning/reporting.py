"""Format training metrics for CSV and TensorBoard reporting."""

from __future__ import annotations

from teleoperation.learning.dataset import HAND_SIDES


LOSS_NAMES = (
    "total",
    "vec",
    "pos",
    "collision",
    "thumb",
    "tip_distance",
    "thumb2",
)


def _metrics_fieldnames(hand_side="shared") -> list[str]:
    names = [
        "epoch",
        "train_total",
        "train_left_total",
        "train_right_total",
        "val_total",
        "val_left_total",
        "val_right_total",
        "learning_rate",
        "gradient_norm",
        "left_valid",
        "right_valid",
    ]
    if hand_side != "shared":
        other = "right" if hand_side == "left" else "left"
        names = [name for name in names if other not in name]
        names.insert(1, "hand_side")
    return names


def _epoch_metrics_row(epoch, train_stats, val_stats, learning_rate, hand_side="shared") -> dict:
    row = {
        "epoch": epoch,
        "train_total": train_stats["total"],
        "val_total": val_stats["total"],
        "learning_rate": learning_rate,
        "gradient_norm": train_stats["gradient_norm"],
    }
    for side in HAND_SIDES if hand_side == "shared" else (hand_side,):
        row[f"train_{side}_total"] = train_stats[f"{side}_total"]
        row[f"val_{side}_total"] = val_stats[f"{side}_total"]
        row[f"{side}_valid"] = train_stats[f"{side}_valid"]
    if hand_side != "shared":
        row["hand_side"] = hand_side
    return row


def _write_epoch_stats(writer, prefix, stats, epoch, hand_side="shared"):
    for name in LOSS_NAMES:
        tag_name = "total" if name == "total" else name
        writer.add_scalar(f"{prefix}/{tag_name}", stats[name], epoch)
    if prefix == "Train":
        writer.add_scalar("Train/gradient_norm", stats["gradient_norm"], epoch)
    for side in HAND_SIDES if hand_side == "shared" else (hand_side,):
        for name in LOSS_NAMES:
            tag_name = "total" if name == "total" else name
            writer.add_scalar(
                f"{prefix}/{side}_{tag_name}",
                stats[f"{side}_{name}"],
                epoch,
            )
        writer.add_scalar(
            f"{prefix}/{side}_valid",
            stats[f"{side}_valid"],
            epoch,
        )

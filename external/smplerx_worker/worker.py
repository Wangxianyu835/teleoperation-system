"""Length-prefixed stdin/stdout worker for a loaded SMPLer-X adapter."""
from __future__ import annotations

import sys

from teleoperation.inputs.smplerx import PacketError, PacketReader, decode_packet, encode_packet


def run_worker(adapter, *, input_stream=None, output_stream=None) -> int:
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer
    reader = PacketReader()
    while True:
        chunk = input_stream.read(64 * 1024)
        if not chunk:
            return 0
        try:
            packets = reader.feed(chunk)
            for packet in packets:
                message = decode_packet(packet)
                if message["type"] != "frame":
                    raise PacketError("worker accepts frame messages only")
                payload = message["payload"]
                result = adapter.infer(payload["rgb"], payload["frame_id"])
                output_stream.write(encode_packet("result", {"result": result}))
                output_stream.flush()
        except (PacketError, ValueError, TypeError) as error:
            print(f"worker protocol error: {error}", file=sys.stderr, flush=True)
            return 2


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from adapter import SmplerXAdapter, SmplerXAssets

    parser = argparse.ArgumentParser()
    parser.add_argument("--smplerx-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--body-model-dir", type=Path, required=True)
    parser.add_argument("--model", default="s32", choices=("s32",))
    options = parser.parse_args()
    adapter = SmplerXAdapter(SmplerXAssets(options.smplerx_root, options.checkpoint,
                                            options.body_model_dir, options.model))
    adapter.load()
    raise SystemExit(run_worker(adapter))

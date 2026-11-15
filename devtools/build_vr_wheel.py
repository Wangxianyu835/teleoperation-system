"""Repackage upstream AVP metadata to use the sole contrib OpenCV provider.

No upstream Python source is changed. The original PyPI wheel is pinned by
SHA-256; the local version explicitly identifies the packaging customization.
This script downloads/builds files only, and does not install any packages.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
from pathlib import Path
import re
import urllib.request
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_VERSION = "2.51"
LOCAL_VERSION = "2.51+opencvcontrib1"
UPSTREAM_FILENAME = "avp_stream-2.51-py3-none-any.whl"
UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/d1/01/"
    "9986ae6745b20dbcc79a0906d58c61f0b479bf97620b732e03510edfcf1d/"
    + UPSTREAM_FILENAME
)
UPSTREAM_SHA256 = "03d19f56ca3d5a2013a0ae353ae68df8f898d1ae3c5567b49419b47421bc2412"
LOCAL_FILENAME = f"avp_stream-{LOCAL_VERSION}-py3-none-any.whl"


def build_wheel(source: Path, output_directory: Path) -> Path:
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != UPSTREAM_SHA256:
        raise ValueError("Upstream wheel SHA-256 mismatch; refusing to repackage")
    old_info = f"avp_stream-{UPSTREAM_VERSION}.dist-info/"
    new_info = f"avp_stream-{LOCAL_VERSION}.dist-info/"
    entries = {}
    with ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            if name.endswith("/") or name == old_info + "RECORD":
                continue
            if name.endswith(("RECORD.jws", "RECORD.p7s")):
                raise ValueError("Signed wheel requires a separate signing procedure")
            content = archive.read(name)
            target = new_info + name[len(old_info):] if name.startswith(old_info) else name
            if name == old_info + "METADATA":
                metadata = content.decode("utf-8")
                metadata, versions = re.subn(
                    r"(?m)^Version: 2\.51$", f"Version: {LOCAL_VERSION}", metadata,
                )
                metadata, requirements = re.subn(
                    r"(?m)^Requires-Dist: opencv-python$",
                    "Requires-Dist: opencv-contrib-python==4.11.0.86", metadata,
                )
                if versions != 1 or requirements != 1:
                    raise ValueError("Unexpected upstream metadata; refusing to guess")
                content = metadata.encode("utf-8")
            entries[target] = content
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, content in sorted(entries.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, len(content)))
    writer.writerow((new_info + "RECORD", "", ""))
    entries[new_info + "RECORD"] = record.getvalue().encode()
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / LOCAL_FILENAME
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(entries.items()):
            archive.writestr(name, content)
    # Check the artifact before returning it, including every upstream code file.
    with ZipFile(source) as original, ZipFile(output) as local:
        for name in original.namelist():
            if name.startswith("avp_stream/"):
                if original.read(name) != local.read(name):
                    raise ValueError(f"Upstream code changed unexpectedly: {name}")
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Already downloaded upstream 2.51 wheel")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/vr_environment")
    args = parser.parse_args(argv)
    source = args.source or args.output_dir / UPSTREAM_FILENAME
    if not source.exists():
        if args.source is not None:
            parser.error(f"Source wheel does not exist: {source}")
        source.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {UPSTREAM_URL}", flush=True)
        with urllib.request.urlopen(UPSTREAM_URL, timeout=60) as response:
            source.write_bytes(response.read())
    output = build_wheel(source, args.output_dir)
    print(f"Built {output}")
    print("Only distribution version and OpenCV requirement changed; AVP source is identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

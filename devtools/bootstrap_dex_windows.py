"""Package official dex 0.5.0 for Windows Conda Pinocchio, without code edits.

PyPI pin has no Windows wheel. Conda's pinocchio supplies the actual binding,
but has no pip 'pin' distribution. Make the dependency platform conditional,
label the local wheel, and verify the real binding separately (never fake pin).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import shutil
import urllib.request


def bundle_runtime(cache, runtime):
    """Copy verified Conda package payloads, without invoking prefix transactions.

    The Python interpreter and NumPy come from the independent project venv.
    Keep the exact package URLs/checksums so the runtime can be reconstructed.
    """
    runtime.mkdir(parents=True, exist_ok=True)
    records = []
    def copy_payload(source, destination):
        if Path(destination).exists():
            if Path(source).read_bytes() != Path(destination).read_bytes():
                raise ValueError(f"Conflicting runtime payload: {destination}")
            return str(destination)
        return shutil.copy2(source, destination)
    excluded = {"python", "python_abi", "numpy", "pip", "setuptools", "wheel"}
    for package in sorted(cache.iterdir()):
        metadata = package / "info/repodata_record.json"
        if not metadata.exists():
            continue
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record["name"] in excluded or "mkl" in package.name:
            continue
        copied = False
        for relative in ("Library/bin", "Lib/site-packages", "site-packages"):
            source = package / relative
            if source.is_dir():
                destination = "Lib/site-packages" if relative == "site-packages" else relative
                shutil.copytree(source, runtime / destination, dirs_exist_ok=True,
                                copy_function=copy_payload,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                copied = True
        if copied:
            records.append({key: record[key] for key in ("name", "version", "build", "url", "sha256")})
    assert any(r["name"] == "pinocchio" and r["version"] == "3.3.1" for r in records)
    (runtime / "payload-lock.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Bundled {len(records)} Conda runtime payloads into {runtime}")


def restore_runtime(lock, cache, runtime):
    """Run with Anaconda Python for conda_package_handling, no Conda transaction."""
    from conda_package_handling.api import extract
    records = json.loads(lock.read_text(encoding="utf-8"))
    selected = cache / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    for record in records:
        filename = record["url"].rsplit("/", 1)[1]
        archive = cache / filename
        if not archive.exists():
            urllib.request.urlretrieve(record["url"], archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {filename}")
        directory = selected / f"{record['name']}-{record['version']}-{record['build']}"
        if not (directory / "info/repodata_record.json").exists():
            directory.mkdir(parents=True, exist_ok=True)
            extract(str(archive.resolve()), dest_dir=str(directory.resolve()))
            (directory / "info/repodata_record.json").write_text(json.dumps(record), encoding="utf-8")
    bundle_runtime(selected, runtime)


def build(wheel: Path, runtime: Path, environment: Path):
    version = "0.5.0+winconda"
    old_info, new_info = "dex_retargeting-0.5.0.dist-info", f"dex_retargeting-{version}.dist-info"
    output = wheel.with_name(f"dex_retargeting-{version}-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        files = {name.replace(old_info, new_info): archive.read(name)
                 for name in archive.namelist() if not name.endswith("/RECORD")}
    metadata = files[new_info + "/METADATA"].decode()
    assert "Requires-Dist: pin>=3.3.1" in metadata
    metadata = metadata.replace("Version: 0.5.0\n", f"Version: {version}\n")
    metadata = metadata.replace("Requires-Dist: pin>=3.3.1\n",
                                'Requires-Dist: pin>=3.3.1; sys_platform != "win32"\n')
    files[new_info + "/METADATA"] = metadata.encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((new_info + "/RECORD", "", ""))
    files[new_info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    site = environment / "Lib/site-packages"
    # Keep DLL directory handles alive for Python >=3.8's explicit DLL loader.
    dll_paths = [runtime / "Library/bin", runtime / "DLLs"]
    pth = str((runtime / "Lib/site-packages").resolve()) + "\n"
    pth += "import os; os._dex_dll_handles = [os.add_dll_directory(p) for p in "
    pth += repr([str(p.resolve()) for p in dll_paths]) + " if os.path.isdir(p)]\n"
    (site / "dex_conda_runtime.pth").write_text(pth, encoding="utf-8")
    report = {"official_wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
              "local_wheel_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "adaptation": "Version local label + conditional Windows pin dependency only; package code unchanged",
              "runtime": str(runtime.resolve()), "code_sha256": {n: hashlib.sha256(d).hexdigest()
              for n, d in files.items() if n.startswith("dex_retargeting/") and n.endswith(".py")}}
    output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


def build_avp(wheel: Path):
    """Retain contrib's cv2 provider; preserve every official SDK payload byte."""
    official_sha = "03d19f56ca3d5a2013a0ae353ae68df8f898d1ae3c5567b49419b47421bc2412"
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != official_sha:
        raise ValueError("Expected the verified official avp_stream 2.51 wheel")
    version = "2.51+wincontrib"
    old_info = "avp_stream-2.51.dist-info"
    new_info = f"avp_stream-{version}.dist-info"
    output = wheel.with_name(f"avp_stream-{version}-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        original = {n: archive.read(n) for n in archive.namelist()}
    files = {n.replace(old_info, new_info): data for n, data in original.items()
             if not n.endswith("/RECORD")}
    metadata = files[new_info + "/METADATA"].decode()
    if metadata.count("Requires-Dist: opencv-python\n") != 1:
        raise ValueError("Unexpected official OpenCV dependency")
    metadata = metadata.replace("Version: 2.51\n", f"Version: {version}\n")
    metadata = metadata.replace("Requires-Dist: opencv-python\n",
                                "Requires-Dist: opencv-contrib-python==4.11.0.86\n")
    files[new_info + "/METADATA"] = metadata.encode()
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
        writer.writerow((name, "sha256=" + digest, len(data)))
    writer.writerow((new_info + "/RECORD", "", ""))
    files[new_info + "/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    payload = {n: hashlib.sha256(d).hexdigest() for n, d in original.items()
               if n.startswith("avp_stream/")}
    with zipfile.ZipFile(output) as archive:
        actual = {n: hashlib.sha256(archive.read(n)).hexdigest()
                  for n in archive.namelist() if n.startswith("avp_stream/")}
    if payload != actual:
        raise ValueError("SDK payload changed")
    report = {"official_wheel_sha256": official_sha,
              "local_wheel_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "adaptation": "Local version label and OpenCV provider dependency only; SDK payload unchanged",
              "payload_sha256": payload}
    output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--avp-wheel", type=Path, help="build the optional Vision Pro SDK compatibility wheel")
    parser.add_argument("--bundle-cache", type=Path)
    parser.add_argument("--restore-lock", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--environment", type=Path)
    args = parser.parse_args()
    if args.avp_wheel:
        if any((args.wheel, args.bundle_cache, args.restore_lock, args.runtime, args.environment)):
            parser.error("--avp-wheel must be used alone")
        build_avp(args.avp_wheel)
        raise SystemExit(0)
    if args.runtime is None:
        parser.error("--runtime is required for the dex runtime")
    if args.bundle_cache:
        bundle_runtime(args.bundle_cache, args.runtime)
        raise SystemExit(0)
    if args.restore_lock:
        if args.cache is None:
            parser.error("--restore-lock requires --cache")
        restore_runtime(args.restore_lock, args.cache, args.runtime)
        raise SystemExit(0)
    if args.wheel is None or args.environment is None:
        parser.error("--wheel and --environment are required for wheel construction")
    build(args.wheel, args.runtime, args.environment)

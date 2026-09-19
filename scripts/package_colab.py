"""Package the current training source for Colab, including uncommitted scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    """Write a source-only archive; exclude credentials, environments and run data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/colab/mario-play.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE")]
    for folder, suffixes in (
        ("src", {".py", ".txt"}),
        ("configs", {".yaml"}),
        ("scripts", {".py"}),
    ):
        files.extend(
            path
            for path in (root / folder).rglob("*")
            if path.is_file() and not path.is_symlink() and path.suffix in suffixes
        )
    contents = {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(files)}
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    manifest = {
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr("source_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Packaged {len(contents)} source files: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()

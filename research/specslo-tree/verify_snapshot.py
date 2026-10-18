"""Verify the saved sources without importing or changing the NPU runtime."""

import ast
import hashlib
import json
from pathlib import Path


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    manifest = json.loads((here / "SOURCE_MANIFEST.json").read_text())
    python_files = 0
    for entry in manifest["files"]:
        path = repo / entry["saved"]
        if not path.is_relative_to(here / "snapshot") or path.is_symlink():
            raise ValueError(f"Invalid saved path: {entry['saved']}")
        data = path.read_bytes()
        if len(data) != entry["bytes"]:
            raise ValueError(f"Size changed: {entry['saved']}")
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"SHA256 changed: {entry['saved']}")
        if path.suffix == ".py":
            ast.parse(data, filename=str(path))
            python_files += 1
    print(json.dumps({"files_verified": len(manifest["files"]),
                      "python_syntax_verified": python_files,
                      "NPU_used": False}))


if __name__ == "__main__":
    main()

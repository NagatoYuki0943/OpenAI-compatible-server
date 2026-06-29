from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_wheel(
    project_dir: Path,
    output_dir: Path,
    *,
    clean: bool = True,
) -> Path:
    pyproject = project_dir / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(f"pyproject.toml not found in {project_dir}")

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv was not found in PATH")

    output_dir = output_dir.resolve()
    if clean and output_dir.exists():
        for wheel in output_dir.glob("*.whl"):
            wheel.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(output_dir),
            str(project_dir),
        ],
        check=True,
    )

    wheels = sorted(output_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime)
    if not wheels:
        raise RuntimeError(f"No wheel was generated in {output_dir}")
    return wheels[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the project wheel with uv.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing pyproject.toml (default: current directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dist"),
        help="Wheel output directory (default: dist)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing wheel files in the output directory",
    )
    args = parser.parse_args()

    wheel = build_wheel(
        args.project_dir.resolve(),
        args.out_dir,
        clean=not args.no_clean,
    )
    print(f"Built wheel: {wheel}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

BACKEND_TEMPLATE_NAME = "custom.py"


def backend_template() -> str:
    return (
        files("openai_compatible.backends")
        .joinpath(BACKEND_TEMPLATE_NAME)
        .read_text(encoding="utf-8")
    )


def create_backend_file(destination: Path, *, force: bool = False) -> Path:
    destination = destination.expanduser().resolve()
    mode = "w" if force else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as backend_file:
        backend_file.write(backend_template())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an editable custom model backend file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("custom_backend.py"),
        help="Destination path (default: custom_backend.py in the current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists",
    )
    args = parser.parse_args()

    try:
        destination = create_backend_file(args.output, force=args.force)
    except FileExistsError:
        parser.error(f"{args.output} already exists; use --force to overwrite it")
    print(f"Created custom backend file: {destination}")
    print("Set MODEL_BACKEND_CLASS=./custom_backend.py:CustomModelBackend in .env")


if __name__ == "__main__":
    main()

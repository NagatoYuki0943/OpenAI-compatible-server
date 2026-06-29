from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path

ENV_TEMPLATE_NAME = "env.example"


def env_template() -> str:
    return files("openai_compatible_server").joinpath(ENV_TEMPLATE_NAME).read_text(encoding="utf-8")


def create_env_file(destination: Path, *, force: bool = False) -> Path:
    destination = destination.expanduser().resolve()
    mode = "w" if force else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as env_file:
        env_file.write(env_template())
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an OpenAI-compatible server .env file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".env"),
        help="Destination path (default: .env in the current directory)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists",
    )
    args = parser.parse_args()

    try:
        destination = create_env_file(args.output, force=args.force)
    except FileExistsError:
        parser.error(f"{args.output} already exists; use --force to overwrite it")
    print(f"Created environment file: {destination}")


if __name__ == "__main__":
    main()

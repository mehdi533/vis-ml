from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute a notebook with nbclient and save the executed copy.")
    parser.add_argument("--notebook", required=True, help="Path to the input notebook.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for the executed notebook. Defaults to <notebook_stem>_executed.ipynb next to the input notebook.",
    )
    parser.add_argument("--timeout", type=int, default=3600, help="Per-cell timeout in seconds.")
    parser.add_argument(
        "--kernel-name",
        default="python3",
        help="Kernel name to use for execution. Defaults to python3.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notebook_path = Path(args.notebook).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else notebook_path.with_name(f"{notebook_path.stem}_executed.ipynb")
    )

    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=int(args.timeout),
        kernel_name=str(args.kernel_name),
        record_timing=True,
        allow_errors=False,
    )
    client.execute()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)

    print(output_path)


if __name__ == "__main__":
    main()

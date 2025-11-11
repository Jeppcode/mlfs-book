#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd


def convert_one(input_path: Path, output_path: Path, verbose: bool = True) -> None:
    """
    Convert an AQICN daily .txt export to a clean CSV with header:
    date,min,max,median,q1,q3,stdev,count

    - Skips comment lines starting with '#'
    - Parses the 'date' column as datetime
    - Writes comma-separated CSV without index
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Read CSV while skipping comment lines beginning with '#'
    # The AQICN daily export already uses comma delimiter and a proper header row.
    df = pd.read_csv(input_path, comment="#", parse_dates=["date"])

    # Basic validation of expected columns
    expected_cols = ["date", "min", "max", "median", "q1", "q3", "stdev", "count"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{input_path} is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Ensure deterministic column order
    df = df[expected_cols]

    # Write out as CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    if verbose:
        print(f"Wrote CSV: {output_path} (rows={len(df)})")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert AQICN daily .txt exports to CSV with standard header."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .txt files to convert. You can pass multiple files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory. Defaults to each file's directory.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else None
    exit_code = 0

    for inp in args.inputs:
        in_path = Path(inp)
        try:
            if output_dir:
                out_path = output_dir / (in_path.stem + ".csv")
            else:
                out_path = in_path.with_suffix(".csv")
            convert_one(in_path, out_path)
        except Exception as e:
            print(f"Error converting {in_path}: {e}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))



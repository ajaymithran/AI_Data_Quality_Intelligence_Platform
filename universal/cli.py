"""Command-line interface for the universal scanner.

Usage::

    python -m universal path/to/file.csv                 # print summary
    python -m universal file.csv --format json           # full JSON to stdout
    python -m universal file.csv --save out_dir          # write all report files
    python -m universal file.csv --fail-on critical      # non-zero exit code

The source CSV is never modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .report import render
from .scanner import scan_csv


def _print_summary(report) -> None:
    d = report.to_dict()
    print(f"Scan {d['scan_id']} - {d['filename']}")
    print(f"  {d['rows']} rows x {len(d['columns'])} columns")
    sev = d["severity_counts"]
    print(f"  findings: {d['finding_total']} "
          f"(critical={sev['CRITICAL']}, warning={sev['WARNING']}, info={sev['INFO']})")
    print("  dimensions:")
    for k, v in d["dimensions"].items():
        bar = "#" * int(v / 5)
        print(f"    {k:<14} {v:>6.1f}  {bar}")
    print("  columns:")
    for c in d["columns"]:
        flags = []
        if c["missing"]:
            flags.append(f"missing={c['missing']}")
        if c["blank"]:
            flags.append(f"blank={c['blank']}")
        if c["placeholder"]:
            flags.append(f"placeholder={c['placeholder']}")
        if c["mixed_types"]:
            flags.append(f"mixed={list(c['mixed_types'])}")
        extra = ("  [" + ", ".join(flags) + "]") if flags else ""
        print(f"    {c['name']:<24} {c['inferred_type']:<10} "
              f"unique={c['unique']}/{c['non_null']}{extra}")
    worst = sorted((f for f in d["findings"] if f["severity"] == "CRITICAL"),
                   key=lambda f: (f["column"], f["row"] or 0))[:15]
    if worst:
        print("  top critical findings:")
        for f in worst:
            loc = f"row {f['row']}" if f["row"] else "column-level"
            print(f"    [{f['rule_id']}] {f['column']} {loc}: {f['message']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m universal",
                                 description="Universal CSV data-quality scanner (read-only)")
    ap.add_argument("csv", help="path to the CSV file to scan")
    ap.add_argument("--format", choices=["summary", "json", "csv", "html"],
                    default="summary", help="what to print to stdout (default: summary)")
    ap.add_argument("--save", metavar="DIR",
                    help="write <name>_quality_report.{json,csv,html} into DIR")
    ap.add_argument("--fail-on", choices=["critical", "warning", "never"],
                    default="never", help="exit non-zero when findings at this level exist")
    ap.add_argument("--row-evidence-cap", type=int, default=None,
                    help="max row-level findings kept per rule+column (default 25)")
    args = ap.parse_args(argv)

    path = Path(args.csv)
    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 2

    cfg = {}
    if args.row_evidence_cap is not None:
        cfg["row_evidence_cap"] = args.row_evidence_cap

    try:
        report = scan_csv(path, config=cfg)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.format == "summary":
        _print_summary(report)
    else:
        payload, _, _ = render(report, args.format)
        sys.stdout.buffer.write(payload)

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        for fmt in ("json", "csv", "html"):
            payload, _, name = render(report, fmt)
            (out / name).write_bytes(payload)
        print(f"\nreports written to {out.resolve()}/")

    sev = report.severity_counts
    if args.fail_on == "critical" and sev["CRITICAL"]:
        return 1
    if args.fail_on == "warning" and (sev["CRITICAL"] or sev["WARNING"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

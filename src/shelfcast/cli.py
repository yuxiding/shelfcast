import argparse
import json
from pathlib import Path

from .data import prepare
from .experiment import run


def main():
    parser = argparse.ArgumentParser(description="Reproduce ShelfCast's real-data benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    reproduce = sub.add_parser("reproduce")
    reproduce.add_argument("--panel", type=Path, default=Path("data/weekly_sales.csv"))
    reproduce.add_argument("--output", type=Path, default=Path("outputs"))
    reproduce.add_argument("--from-source", action="store_true", help="Download and rebuild the panel from the original UCI workbook")
    args = parser.parse_args()
    if args.from_source:
        panel, audit, names = prepare(Path("data/raw"))
        args.output.mkdir(parents=True, exist_ok=True)
        args.panel = args.output / "prepared_weekly_sales.csv"
        panel.to_csv(args.panel, index=False)
        (args.output / "data_audit.json").write_text(json.dumps(audit, indent=2))
    else:
        audit_file, names_file = args.panel.parent / "audit.json", args.panel.parent / "products.json"
        audit = json.loads(audit_file.read_text()) if audit_file.exists() else None
        names = json.loads(names_file.read_text()) if names_file.exists() else None
    result = run(args.panel, args.output, audit, names)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

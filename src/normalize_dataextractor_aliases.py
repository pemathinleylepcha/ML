from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from dataextractor_contract import SYMBOL_DIR_ALIASES


def normalize_alias_dirs(root: Path, apply: bool = False) -> list[dict]:
    actions: list[dict] = []
    for year_dir in sorted([p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]):
        for quarter_dir in sorted([q for q in year_dir.iterdir() if q.is_dir() and q.name.startswith("Q")]):
            for canonical_symbol, aliases in SYMBOL_DIR_ALIASES.items():
                canonical_dir = quarter_dir / canonical_symbol
                for alias in aliases:
                    alias_dir = quarter_dir / alias
                    if not alias_dir.is_dir():
                        continue

                    action = {
                        "year": year_dir.name,
                        "quarter": quarter_dir.name,
                        "alias": alias,
                        "canonical": canonical_symbol,
                        "alias_dir": str(alias_dir),
                        "canonical_dir": str(canonical_dir),
                        "operation": "merge" if canonical_dir.exists() else "rename",
                    }
                    actions.append(action)

                    if not apply:
                        continue

                    if canonical_dir.exists():
                        canonical_dir.mkdir(parents=True, exist_ok=True)
                        for item in alias_dir.iterdir():
                            target = canonical_dir / item.name
                            if item.is_dir():
                                if target.exists():
                                    raise RuntimeError(f"Refusing to overwrite existing directory {target}")
                                shutil.move(str(item), str(target))
                            else:
                                if target.exists():
                                    # Preserve existing canonical file; alias trees are historical fallbacks.
                                    continue
                                shutil.move(str(item), str(target))
                        alias_dir.rmdir()
                    else:
                        alias_dir.rename(canonical_dir)
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize known alias symbol directories in DataExtractor.")
    parser.add_argument("--root", default="data/DataExtractor", help="DataExtractor root")
    parser.add_argument("--apply", action="store_true", help="Apply the rename/merge operations")
    args = parser.parse_args()

    actions = normalize_alias_dirs(Path(args.root), apply=args.apply)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"{mode}: {len(actions)} alias operations found")
    for action in actions:
        print(
            f"{action['year']}/{action['quarter']}: {action['alias']} -> {action['canonical']} "
            f"({action['operation']})"
        )


if __name__ == "__main__":
    main()

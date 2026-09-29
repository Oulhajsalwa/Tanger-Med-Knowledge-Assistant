

from __future__ import annotations

import argparse
import sys

from app.ingestion.pipeline import run_ingestion
from app.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion",
        description="Indexe les rapports annuels et RSE de Tanger Med.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconstruit le vector store et l'index BM25 en ignorant le manifeste.",
    )
    args = parser.parse_args()

    report = run_ingestion(force=args.force)

    if report.processed_files:
        print("\nFichiers traités :")
        for f in report.processed_files:
            print(f"  - {f}")
    if report.skipped_unchanged_files:
        print("\nFichiers inchangés (ignorés) :")
        for f in report.skipped_unchanged_files:
            print(f"  - {f}")
    if report.skipped_duplicate_files:
        print("\nFichiers dupliqués (ignorés) :")
        for f in report.skipped_duplicate_files:
            print(f"  - {f}")
    if report.failed_files:
        print("\nFichiers en échec :")
        for f in report.failed_files:
            print(f"  - {f}")

    print(f"\n{report.summary()}")
    return 0 if not report.failed_files else 1


if __name__ == "__main__":
    sys.exit(main())

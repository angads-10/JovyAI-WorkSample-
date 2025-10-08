import csv
import os
from typing import Dict, Iterable, Any


class MetricsLogger:
    """Simple CSV and console metrics logger.

    - Appends rows to a CSV file, creating it with headers on first write.
    - Prints a concise line to console for each log.
    - Provides helpers for pretty headers/summaries.
    """

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self._fieldnames: Iterable[str] | None = None
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def log(self, row: Dict[str, Any]) -> None:
        # Lazily set fieldnames in a deterministic order
        if self._fieldnames is None:
            self._fieldnames = list(sorted(row.keys()))
            write_header = True
        else:
            write_header = not os.path.exists(self.csv_path)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, None) for k in self._fieldnames})

        # Console print
        ordered = {k: row[k] for k in sorted(row.keys())}
        print(" | ".join(f"{k}={ordered[k]}" for k in ordered))

    def print_header(self, text: str) -> None:
        print("=" * 80)
        print(text)
        print("=" * 80)

    def print_summary(self, final_row: Dict[str, Any]) -> None:
        print("-" * 80)
        print("Final KPIs:")
        ordered = {k: final_row[k] for k in sorted(final_row.keys())}
        for k, v in ordered.items():
            print(f"  {k}: {v}")
        print("-" * 80)



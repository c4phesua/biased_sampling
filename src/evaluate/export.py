"""Export benchmark results to CSV file"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Union


def export_to_csv(results, file_path: Union[str, Path]):
    if not results:
        raise ValueError("No benchmark results to export.")

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = list(results[0].keys())

    with file_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmark results exported to: {str(file_path)}")
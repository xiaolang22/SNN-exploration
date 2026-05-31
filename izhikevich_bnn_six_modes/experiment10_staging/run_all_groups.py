from __future__ import annotations

import subprocess
import sys
from pathlib import Path


GROUPS = [
    "01_stdp_only",
    "02_stdp_rate_target_ip",
    "03_stdp_spikl_ip",
    "04_spikl_ip_only",
]


def main() -> None:
    root = Path(__file__).parent
    for group in GROUPS:
        group_dir = root / group
        print("=" * 70)
        print(f"Running {group}")
        print("=" * 70)
        subprocess.run([sys.executable, "run.py"], cwd=group_dir, check=True)


if __name__ == "__main__":
    main()

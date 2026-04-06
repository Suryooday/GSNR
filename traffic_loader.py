"""Traffic loader module for optical network simulation."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def load_traffic(file_path: str, default_bit_rate: float = 100.0) -> List[Dict[str, object]]:
    """Load a network traffic demand dataset from a CSV or custom text file.

    Args:
        file_path: Path to the traffic dataset file.
        default_bit_rate: Base bandwidth assignment if none is provided.

    Returns:
        List of dictionaries with keys compatible with `traffic_engine`:
        [{"src": source, "dst": destination, "bit_rate": demand, "time": timestamp}]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic dataset not found: {file_path}")

    dataset: List[Dict[str, object]] = []

    # Detect file type
    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = None

            # Look for headers if present
            sample_line = f.readline()
            f.seek(0)
            if "src" in sample_line.lower() or "source" in sample_line.lower():
                reader = csv.DictReader(f)
                for row in reader:
                    src = str(row.get("src", row.get("source", "")))
                    dst = str(row.get("dst", row.get("destination", "")))
                    demand = float(row.get("demand", row.get("bit_rate", default_bit_rate)))
                    time = float(row.get("time", row.get("timestamp", 0.0)))
                    holding_time = float(row.get("holding_time", 1.0))
                    
                    dataset.append({
                        "src": src,
                        "dst": dst,
                        "bit_rate": demand,
                        "time": time,
                        "holding_time": holding_time
                    })
            else:
                # Assume raw columns: src, dst, demand, [timestamp]
                for row in reader:
                    if len(row) < 2:
                        continue
                        
                    src = str(row[0]).strip()
                    dst = str(row[1]).strip()
                    demand = float(row[2]) if len(row) > 2 else default_bit_rate
                    time = float(row[3]) if len(row) > 3 else 0.0
                    
                    dataset.append({
                        "src": src,
                        "dst": dst,
                        "bit_rate": demand,
                        "time": time
                    })
    else:
        # Standard .TXT whitespace-separated parsing
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                # ignore comments and empty lines
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    src = str(parts[0])
                    dst = str(parts[1])
                    demand = float(parts[2]) if len(parts) > 2 else default_bit_rate
                    time = float(parts[3]) if len(parts) > 3 else 0.0
                    dataset.append({
                        "src": src,
                        "dst": dst,
                        "bit_rate": demand,
                        "time": time
                    })

    logger.info("Loaded %d traffic demands from %s", len(dataset), path.name)
    return dataset


def get_traffic_tuples(file_path: str) -> List[Tuple[str, str, float]]:
    """Convenience wrapper to return raw tuples (src, dst, demand) exactly as requested."""
    data = load_traffic(file_path)
    return [(str(req["src"]), str(req["dst"]), float(req["bit_rate"])) for req in data]

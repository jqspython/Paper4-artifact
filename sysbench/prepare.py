"""Select proposed fragment IDs and HPA utilization from the frozen exploration only.

No demand arrays, confirmation applications or system outcomes are read.
"""
import csv
import hashlib
import io
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "v0.7-v2-explore"
SEED = 20261022


def selections():
    raw = subprocess.check_output(["git", "show", REFERENCE+":results/v2-lite-summary.csv"], cwd=ROOT)
    rows = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    configurations = {}
    for row in rows:
        if row["stage"] != "explore" or row["phase"] != "test" or row["role"] != "locked" or row["base"] != "HPA" or row["wrapped"] != "0":
            continue
        if row["status"] != "OK":
            raise ValueError("Unavailable frozen HPA base")
        identity = row["source"], row["app"]
        value = float(row["parameter"])
        if identity in configurations and configurations[identity] != value:
            raise ValueError("Frozen base varies across cases")
        configurations[identity] = value
    if len(configurations) != 150:
        raise ValueError("Expected exactly the original 150 exploratory applications")
    chosen = []
    for vendor in ("Huawei", "Azure"):
        pool = [key for key in configurations if (key[0] == "azure2019") == (vendor == "Azure")]
        pool.sort(key=lambda key: hashlib.sha256(f"v3-sys|{SEED}|{key[0]}|{key[1]}".encode()).hexdigest())
        for i, (source, app) in enumerate(pool[:6]):
            chosen.append(dict(role="formal-proposed" if i < 3 else "shakedown-reserve", vendor=vendor,
                               source=source, app=app, vendor_hash_order=i+1, ticks=240,
                               start_rule="first_240_minutes_of_frozen_exploration_test", fit_rule="original_fitting_prefix",
                               reference=REFERENCE, seed=SEED))
    assert len({(r["source"], r["app"]) for r in chosen}) == 12
    return statistics.median(configurations.values()), chosen


if __name__ == "__main__":
    utilization, rows = selections()
    target = ROOT / "results/v3-sys-fragment-ids.csv"
    with target.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("Frozen exploratory HPA target median:", utilization, "; proposed IDs:", target.relative_to(ROOT))
    print("No demand arrays read; formal selection remains a draft until protocol freeze")

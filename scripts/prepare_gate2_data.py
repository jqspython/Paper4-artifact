"""Build Gate-2 minute demand arrays; selection reads the first five days only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import codecs
import csv
import hashlib
import io
from pathlib import Path
import re
import tarfile
import urllib.request
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/gate2"
OUT = ROOT / "data/processed/gate2"
M = 1440
PREFIX = 5 * M
FIT_END, CAL_END, VAL_END = 3 * M, 5 * M, 8 * M
SOURCES = {
    "huawei_private2023": (
        "huawei2023_requests_minute.zip",
        "https://drive.usercontent.google.com/download?id=1W9dQvqRGylfUeHJcYqWwJrpYHIsYCVq_&export=download&confirm=t",
        40197213, 28, 60),
    "huawei_public2023": (
        "huawei_public2023_requests_minute.zip",
        "https://drive.google.com/uc?export=download&id=1RfkSxt9TddpOHiybEJFwUXfWqpsJyqSK",
        9738517, 0, 25),
    "azure2019": (
        "azure2019.tar.xz",
        "https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz",
        142968140, 1, 14),
}


def rank_id(unit):
    return hashlib.sha256(("42:" + str(unit)).encode()).hexdigest()


def choose(stats, count=50):
    """stats maps IDs to (prefix sum, nonzero minutes, active days, fit peak)."""
    eligible = [u for u, (_, nz, days, peak) in stats.items()
                if nz >= 0.01 * PREFIX and days >= 2 and peak > 0]
    selected = sorted(eligible, key=lambda u: (rank_id(u), str(u)))[:count]
    if len(selected) < count:
        raise ValueError(f"Only {len(eligible)} eligible units; need {count}; do not weaken after viewing results")
    return selected, set(eligible)


def write_data(family, demand, observed, selected, stats, eligible):
    if demand.shape != observed.shape or demand.shape[0] != 50:
        raise ValueError("Unexpected demand/mask shape")
    if not np.isfinite(demand).all() or (demand < 0).any():
        raise ValueError("Demand must be finite and nonnegative after declared zero insertion")
    storage = demand.astype(np.float32)
    if not np.array_equal(storage.astype(np.float64), demand):
        raise ValueError("Float32 would change invocation counts; preserve source precision before execution")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{family}.npz", demand=storage,
                        observed=observed, unit_id=np.asarray(selected, dtype="U64"),
                        fit_end=FIT_END, cal_end=CAL_END, val_end=VAL_END,
                        first_day=SOURCES[family][3])
    with (OUT / f"{family}-units.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unit_id", "selection_rank", "prefix_mean", "prefix_nonzero_fraction", "prefix_active_days", "fit_peak", "prefix_peak", "prefix_observed_fraction"])
        for i, unit in enumerate(selected):
            total, nz, days, fit_peak = stats[unit]
            w.writerow([unit, i + 1, total / PREFIX, nz / PREFIX, days, fit_peak,
                        float(demand[i, :PREFIX].max()), float(observed[i, :PREFIX].mean())])
    # A compact eligibility ledger records who could have been selected, using prefix information only.
    with (OUT / f"{family}-eligibility.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["unit_id", "prefix_mean", "prefix_nonzero_fraction", "prefix_active_days", "fit_peak", "eligible", "selected"])
        for unit in sorted(stats):
            total, nz, days, fit_peak = stats[unit]
            w.writerow([unit, total / PREFIX, nz / PREFIX, days, fit_peak, unit in eligible, unit in selected])
    print(f"{family}: population={len(stats)} eligible={len(eligible)} selected={len(selected)} "
          f"minutes={demand.shape[1]} test_minutes={demand.shape[1]-VAL_END}", flush=True)


def huawei(family):
    filename, _, _, first, last = SOURCES[family]
    with zipfile.ZipFile(RAW / filename) as z:
        members = {}
        for name in z.namelist():
            if "__MACOSX" in name:
                continue
            match = re.search(r"(?:^|/)requests_minute/day_(\d+)\.csv$", name)
            if match:
                day = int(match[1])
                if day in members:
                    raise ValueError(f"Duplicate day {day}")
                members[day] = name
        if not set(range(first, last + 1)).issubset(members):
            raise ValueError(f"Missing requested days for {family}")

        def load(day, ids=None):
            with z.open(members[day]) as handle:
                r = csv.reader(io.TextIOWrapper(handle, encoding="utf-8-sig", newline=""))
                header = next(r)
                if header[:2] != ["day", "time"]:
                    raise ValueError(f"Unexpected Huawei schema: {header[:2]}")
                names = header[2:]
                positions = list(range(2, len(header))) if ids is None else [header.index(u) for u in ids]
                values, masks, clocks = [], [], []
                for row in r:
                    if len(row) != len(header):
                        raise ValueError("Inconsistent Huawei row width")
                    if int(row[0]) != day:
                        raise ValueError("Huawei day column disagrees with file name")
                    clocks.append(float(row[1]))
                    cells = [row[i] for i in positions]
                    mask = [s != "" and s.lower() != "nan" for s in cells]
                    masks.append(mask)
                    values.append([float(s) if present else 0.0 for s, present in zip(cells, mask)])
                if len(values) != M or clocks[0] != day * 86400 or not np.allclose(np.diff(clocks), 60):
                    raise ValueError(f"Day {day} does not have 1440 consecutive minute timestamps")
                return np.asarray(values, dtype=np.float64).T, np.asarray(masks, dtype=bool).T, names

        totals = nonzero = active = None
        names0 = None
        prefix_blocks = []
        prefix_masks = []
        for day in range(first, first + 5):
            block, mask, names = load(day)
            if names0 is None:
                names0 = names
                totals = np.zeros(len(names)); nonzero = np.zeros(len(names), dtype=int)
                active = np.zeros(len(names), dtype=int)
            if names != names0:
                raise ValueError("Huawei function IDs/order changed within prefix")
            totals += block.sum(1)
            nonzero += (block > 0).sum(1)
            active += (block > 0).any(1)
            prefix_blocks.append(block); prefix_masks.append(mask)
        fit_peak = np.maximum.reduce([b.max(1) for b in prefix_blocks[:3]])
        stats = {u: (float(totals[i]), int(nonzero[i]), int(active[i]), float(fit_peak[i]))
                 for i, u in enumerate(names0)}
        selected, eligible = choose(stats)
        positions = [names0.index(u) for u in selected]
        demand = np.zeros((50, (last-first+1)*M), dtype=np.float64)
        observed = np.zeros_like(demand, dtype=bool)
        for k in range(5):
            demand[:, k*M:(k+1)*M] = prefix_blocks[k][positions]
            observed[:, k*M:(k+1)*M] = prefix_masks[k][positions]
        del prefix_blocks, prefix_masks
        for day in range(first+5, last+1):
            block, mask, _ = load(day, selected)
            k = day-first
            demand[:, k*M:(k+1)*M] = block
            observed[:, k*M:(k+1)*M] = mask
        write_data(family, demand, observed, selected, stats, eligible)


def azure_blocks(days, selected=None):
    # Native seekable LZMA reads avoid quadratic _Stream buffer slicing on sparse CSVs.
    with tarfile.open(RAW / SOURCES["azure2019"][0], "r:xz") as tar:
        for member in tar:
            match = re.search(r"(?:^|/)invocations_per_function_md\.anon\.d(\d+)\.csv$", member.name)
            if not match or int(match[1]) not in days:
                continue
            day = int(match[1])
            block = {}
            with tar.extractfile(member) as binary:
                # codecs also accepts TarFile streams without seekable() on Python 3.11.
                text = codecs.getreader("utf-8-sig")(binary)
                reader = csv.reader(text)
                header = next(reader)
                app_pos = header.index("HashApp")
                first_minute = header.index("1")
                if header[first_minute:] != [str(i) for i in range(1, M+1)]:
                    raise ValueError("Unexpected Azure minute schema")
                if selected is not None:
                    # Official hashed ID fields contain no commas/newlines; skip other apps before 1440-field tokenization.
                    lines = (line for line in text
                             if line.split(",", app_pos+1)[app_pos].strip('"') in selected)
                    reader = csv.reader(lines)
                for row in reader:
                    app = row[app_pos]
                    if selected is not None and app not in selected:
                        continue
                    values = np.fromstring(",".join(row[first_minute:]), sep=",", dtype=np.float64)
                    if values.size != M or not np.isfinite(values).all() or (values < 0).any():
                        raise ValueError("Invalid Azure invocation counts")
                    if app in block:
                        block[app] += values
                    else:
                        block[app] = values
            print(f"azure2019 read day={day} apps={len(block)}", flush=True)
            yield day, block


def azure():
    totals = defaultdict(float); nonzero = defaultdict(int); active = defaultdict(int)
    fit_peak = defaultdict(float)
    prefix_blocks = {}
    for day, block in azure_blocks(set(range(1, 6))):
        prefix_blocks[day] = block
        for unit, values in block.items():
            totals[unit] += values.sum()
            nonzero[unit] += np.count_nonzero(values)
            active[unit] += int((values > 0).any())
            if day <= 3:
                fit_peak[unit] = max(fit_peak[unit], float(values.max()))
    if set(prefix_blocks) != set(range(1, 6)):
        raise ValueError("Azure prefix incomplete")
    stats = {u: (float(v), int(nonzero[u]), int(active[u]), fit_peak[u]) for u, v in totals.items()}
    selected, eligible = choose(stats)
    demand = np.zeros((50, 14*M), dtype=np.float64)
    observed = np.zeros_like(demand, dtype=bool)
    for day, block in prefix_blocks.items():
        for i, unit in enumerate(selected):
            if unit in block:
                demand[i, (day-1)*M:day*M] = block[unit]
                observed[i, (day-1)*M:day*M] = True
    del prefix_blocks
    seen = set(range(1, 6))
    for day, block in azure_blocks(set(range(6, 15)), set(selected)):
        seen.add(day)
        for i, unit in enumerate(selected):
            if unit in block:
                demand[i, (day-1)*M:day*M] = block[unit]
                observed[i, (day-1)*M:day*M] = True
    if seen != set(range(1, 15)):
        raise ValueError("Azure day set incomplete")
    write_data("azure2019", demand, observed, selected, stats, eligible)


def download(family):
    filename, url, expected, _, _ = SOURCES[family]
    path = RAW / filename
    RAW.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected:
        return
    partial = path.with_suffix(path.suffix + ".part")
    if family == "azure2019":
        # The official whole-asset response stalled here; bounded Range requests worked.
        with urllib.request.urlopen(urllib.request.Request(url + "?gate2=range", headers={"Range": "bytes=0-0"}), timeout=30) as r:
            direct = r.geturl()
        chunk_size = 4 * 1024 * 1024
        def part(index):
            lo, hi = index * chunk_size, min(expected, (index + 1) * chunk_size) - 1
            with urllib.request.urlopen(urllib.request.Request(direct, headers={"Range": f"bytes={lo}-{hi}"}), timeout=30) as r:
                if r.status != 206:
                    raise RuntimeError("Server did not honor the bounded Range request")
                data = r.read()
            if len(data) != hi-lo+1:
                raise ValueError("Incomplete archive chunk")
            return data
        with partial.open("wb") as f, ThreadPoolExecutor(max_workers=4) as pool:
            for data in pool.map(part, range((expected + chunk_size - 1) // chunk_size)):
                f.write(data)
        partial.replace(path)
        return
    with urllib.request.urlopen(url, timeout=60) as r:
        if r.headers.get("Content-Type", "").startswith("text/html"):
            raise RuntimeError("Download returned HTML rather than trace archive")
        with partial.open("wb") as f:
            while chunk := r.read(65536):
                f.write(chunk)
    if partial.stat().st_size != expected:
        raise ValueError("Archive size differs from the official asset inspected at protocol preparation")
    partial.replace(path)


def self_check():
    stats = {str(i): (100.0, 72, 2, 1.0) for i in range(60)}
    selected, eligible = choose(stats)
    assert len(selected) == 50 and len(eligible) == 60
    assert selected == sorted(stats, key=lambda u: (rank_id(u), str(u)))[:50]
    stats["0"] = (100, 71, 2, 1)
    stats["1"] = (100, 72, 1, 1)
    stats["2"] = (100, 72, 2, 0)
    assert "0" not in choose(stats)[1] and "1" not in choose(stats)[1]
    assert "2" not in choose(stats)[1]
    assert (FIT_END, CAL_END, VAL_END) == (4320, 7200, 11520)
    payload = b"HashApp,1,2\na,3,4\n"
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w") as tar:
        entry = tarfile.TarInfo("tiny.csv"); entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    memory.seek(0)
    with tarfile.open(fileobj=memory, mode="r|") as tar:
        member = next(iter(tar))
        with tar.extractfile(member) as stream:
            rows = list(csv.reader(codecs.getreader("utf-8-sig")(stream)))
            assert rows == [["HashApp", "1", "2"], ["a", "3", "4"]]
    print("self_check=PASS")


def write_id_manifest():
    """Share selection identity without redistributing numerical demand derivatives."""
    paths = [(family, OUT / f"{family}-units.csv") for family in SOURCES]
    if not all(path.exists() for _, path in paths):
        return
    with (OUT / "selected-unit-ids.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["trace", "selection_rank", "unit_id"])
        for family, path in paths:
            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            if len(rows) != 50:
                raise ValueError("Selection manifest requires exactly 50 IDs per trace")
            writer.writerows((family, row["selection_rank"], row["unit_id"]) for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=[*SOURCES, "all"], default="all")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check(); return
    for family in SOURCES if args.family == "all" else [args.family]:
        if args.download:
            download(family)
        if family.startswith("huawei"):
            huawei(family)
        else:
            azure()
    write_id_manifest()


if __name__ == "__main__":
    main()

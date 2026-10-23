"""Prepare the frozen confirmation cohort in identity-first, gated stages.

Neither stage downloads data or re-screens eligibility. No archive/NPZ demand is
opened by identities; demands validates the entire cohort before any raw reader.
"""

import argparse
import csv
import hashlib
import io
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_gate2_data as legacy
from v2_lite import load_config, require_stage_gate, validate_manifest

SOURCES = ("huawei_private2023", "huawei_public2023", "azure2019")
POOL_COUNTS = dict(zip(SOURCES, (107, 543, 12643)))
CONFIRM_COUNTS = dict(zip(SOURCES, (57, 80, 80)))
DESIGN_SHA = "4373a4ed66a99d4e26b5d81743e308229af2ea13"
PREFIX = "20261020-v2-confirm:"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, encoding="utf-8")


def frozen_frame(config):
    """Read only source/ID metadata from the immutable design reference."""
    tag = config["gates"]["design_tag"]
    need(git("rev-parse", tag + "^{commit}").strip() == DESIGN_SHA, "The design identity frame changed")
    result = []
    for name, counts in (("eligible-unit-pool.csv", POOL_COUNTS),
                         ("selected-unit-ids.csv", dict.fromkeys(SOURCES, 50))):
        text = git("show", f"{tag}:data/processed/gate2/{name}")
        groups = {s: [] for s in SOURCES}
        for row in csv.DictReader(io.StringIO(text)):
            source = row.get("source", row.get("trace"))
            need(source in groups and row.get("unit_id"), "Unknown source or empty frozen identity")
            groups[source].append(row["unit_id"])
        need(all(len(ids) == len(set(ids)) == counts[s] for s, ids in groups.items()),
             f"Frozen frame counts/duplicates differ: {name}")
        result.append(groups)
    pool, old = result
    need(all(set(old[s]) <= set(pool[s]) for s in SOURCES), "Old exploration IDs are outside the frozen frame")
    return pool, old


def selected_ids(pool, old, config):
    need(config["selection"]["hash_prefix"] == PREFIX and config["design"]["confirmation_cap_per_trace"] == 80,
         "Confirmation hash/cap differ from the reviewed rule")
    selected = {s: sorted(set(pool[s]) - set(old[s]), key=lambda a: (
        hashlib.sha256((PREFIX + a).encode("utf-8")).digest(), a))[:80] for s in SOURCES}
    need(all(len(selected[s]) == CONFIRM_COUNTS[s] and not set(selected[s]) & set(old[s]) for s in SOURCES),
         "Confirmation requires exactly 57/80/80 source-qualified, nonoverlapping IDs")
    return selected


def output_directory(config):
    output = (ROOT / config["io"]["confirmation_data"]).resolve()
    need(output.is_relative_to(ROOT.resolve()) and output != (ROOT / config["io"]["exploration_data"]).resolve(),
         "Confirmation output must be a separate project-local directory")
    return output


def identities(config, output):
    pool, old = frozen_frame(config)
    selected = selected_ids(pool, old, config)
    expected = [(s, str(i + 1), a) for s in SOURCES for i, a in enumerate(selected[s])]
    path = output / "selected-unit-ids.csv"
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            actual = [(r.get("source", r.get("trace")), r["selection_rank"], r["unit_id"]) for r in csv.DictReader(handle)]
        need(actual == expected, "Existing confirmation manifest differs; refusing replacement")
    else:
        output.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("source", "selection_rank", "unit_id"))
            writer.writerows(expected)
        temporary.replace(path)
    checked = validate_manifest(config, "confirm", directory=output)
    need(all(tuple(selected[s]) == tuple(checked[s]) for s in SOURCES), "Manifest verification changed identity/order")
    return checked


def checked_manifest(config, output):
    # The shared manifest guard precedes even raw archive inventory or opening.
    prepared = validate_manifest(config, "confirm", directory=output)
    pool, old = frozen_frame(config)
    selected = selected_ids(pool, old, config)
    need(all(tuple(prepared[s]) == tuple(selected[s]) and len(prepared[s]) == CONFIRM_COUNTS[s]
             and not set(prepared[s]) & set(old[s]) for s in SOURCES), "Confirmation manifest differs from frozen order/exclusion")
    return prepared


def raw_inventory():
    for source in SOURCES:
        filename, _, expected_size, _, _ = legacy.SOURCES[source]
        path = legacy.RAW / filename
        need(path.is_file() and path.stat().st_size == expected_size,
             f"NOT_PREPARED: missing or different raw archive {filename}; no automatic download")


def huawei_selected(source, selected):
    """The legacy selected-column parser, without its prefix re-screening stage."""
    filename, _, _, first, last = legacy.SOURCES[source]
    demand = np.zeros((len(selected), (last - first + 1) * legacy.M), dtype=np.float64)
    observed = np.zeros(demand.shape, dtype=bool)
    with zipfile.ZipFile(legacy.RAW / filename) as archive:
        members = {}
        for name in archive.namelist():
            if "__MACOSX" in name:
                continue
            match = re.search(r"(?:^|/)requests_minute/day_(\d+)\.csv$", name)
            if match:
                day = int(match[1])
                need(day not in members, "Duplicate Huawei day")
                members[day] = name
        need(set(range(first, last + 1)) <= set(members), "Missing Huawei day")
        for day in range(first, last + 1):
            with archive.open(members[day]) as binary:
                reader = csv.reader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
                header = next(reader)
                need(header[:2] == ["day", "time"] and len(header[2:]) == len(set(header[2:])), "Unexpected Huawei schema")
                need(all(a in header[2:] for a in selected), "Selected Huawei identity absent from a day header")
                positions = [header.index(a) for a in selected]
                values, masks, clocks = [], [], []
                for row in reader:
                    need(len(row) == len(header) and int(row[0]) == day, "Huawei row/day mismatch")
                    clocks.append(float(row[1]))
                    cells = [row[i] for i in positions]
                    mask = [v != "" and v.lower() != "nan" for v in cells]
                    masks.append(mask)
                    values.append([float(v) if ok else 0. for v, ok in zip(cells, mask)])
                need(len(values) == legacy.M and clocks[0] == day * 86400
                     and np.allclose(np.diff(clocks), 60), "Huawei minute clock differs")
                left = (day - first) * legacy.M
                demand[:, left:left + legacy.M] = np.asarray(values, dtype=np.float64).T
                observed[:, left:left + legacy.M] = np.asarray(masks, dtype=bool).T
    return demand, observed


def azure_selected(selected):
    _, _, _, first, last = legacy.SOURCES["azure2019"]
    demand = np.zeros((len(selected), (last - first + 1) * legacy.M), dtype=np.float64)
    observed = np.zeros(demand.shape, dtype=bool)
    seen = set()
    # Reuse the official-app aggregation, selected-line filtering and zero semantics.
    for day, block in legacy.azure_blocks(set(range(first, last + 1)), set(selected)):
        need(day not in seen, "Duplicate Azure day")
        seen.add(day)
        for i, app in enumerate(selected):
            if app in block:
                left = (day - first) * legacy.M
                demand[i, left:left + legacy.M] = block[app]
                observed[i, left:left + legacy.M] = True
    need(seen == set(range(first, last + 1)), "Azure day set incomplete")
    return demand, observed


def save_source(output, source, selected, demand, observed):
    first, last = legacy.SOURCES[source][3:]
    need(demand.shape == observed.shape == (len(selected), (last - first + 1) * legacy.M), "Unexpected selected array shape")
    need(np.isfinite(demand).all() and (demand >= 0).all(), "Demand must remain finite/nonnegative")
    storage = demand.astype(np.float32)
    if not np.array_equal(storage.astype(np.float64), demand):
        storage = demand  # Preserve exact counts rather than round a newly selected unit.
    path = output / f"{source}.npz"
    need(not path.exists(), "Existing confirmation demand is not overwritten")
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, demand=storage, observed=observed.astype(bool),
                            unit_id=np.asarray(selected, dtype=str), fit_end=legacy.FIT_END,
                            cal_end=legacy.CAL_END, val_end=legacy.VAL_END, first_day=first)
    temporary.replace(path)


def prepare(stage, config=None):
    config = load_config() if config is None else config
    require_stage_gate(config, "confirm", experiment=True)
    need(stage in ("identities", "demands"), "Unknown preparation stage")
    output = output_directory(config)
    if stage == "identities":
        return identities(config, output)
    selected = checked_manifest(config, output)
    need(not any((output / f"{s}.npz").exists() for s in SOURCES), "Confirmation arrays already exist; refusing replacement")
    raw_inventory()
    for source in SOURCES:
        demand, observed = (huawei_selected(source, selected[source]) if source.startswith("huawei")
                            else azure_selected(selected[source]))
        save_source(output, source, selected[source], demand, observed)
        print(f"{source}: prepared {len(selected[source])} fixed units, {demand.shape[1]} minutes", flush=True)
    return selected


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("identities", "demands"), required=True)
    args = parser.parse_args(argv)
    prepared = prepare(args.stage)
    print(f"{args.stage}: " + ", ".join(f"{s}={len(prepared[s])}" for s in SOURCES))


if __name__ == "__main__":
    main()

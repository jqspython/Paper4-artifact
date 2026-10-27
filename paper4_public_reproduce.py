"""Regenerate manuscript table data from the seven public aggregate CSVs.

The checker deliberately reports insufficient public statistics instead of
embedding private values.  It performs no experiment and never reads trace data.
"""

import argparse
import csv
from pathlib import Path
import re


FILES = (
    "v2-lite-confirm-endpoints.csv", "v2-lite-confirm-risk.csv",
    "v2-lite-confirm-descriptive.csv", "v1.1-static-reference-summary.csv",
    "v1.1-lambda-diagnostic.csv", "v1.1-azure2021-qualification.csv",
    "v3-sys-closure.csv",
)
CASES = ("nominal", "delay120", "delay180", "delay300", "delay600", "ramp1.5",
         "ramp2", "step1.5", "step2", "flash3x15", "flash5x60", "rate1", "cap8_rate1")
CLASSES = {"U": ("HPA", "Point+margin"), "O": ("PAC-h", "BACC-cal")}


def _read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _one(rows, **keys):
    found = [row for row in rows if all(row.get(key) == value for key, value in keys.items())]
    if len(found) != 1:
        raise ValueError(f"Expected one row for {keys}; found {len(found)}")
    return found[0]


def _fmt(value, scale=1., digits=3):
    return f"{float(value) * scale:.{digits}f}"


def _endpoint(rows, metric, scope="overall", variant="primary"):
    return _one(rows, metric=metric, scope=scope, variant=variant)


def _table3(endpoints):
    result = []
    for label, metric, scale in (("P1 protection (pp)", "p1_delta", 100.),
                                 ("U median cost guard", "p1_cost_ratio", 1.),
                                 ("P2 O median cost ratio", "p2_cost_ratio", 1.),
                                 ("O compliance change (pp)", "p2_delta", 100.),
                                 ("P3 compliance (%)", "p3_compliance", 100.)):
        overall = _endpoint(endpoints, metric)
        result.append({"Metric": label,
            "Overall [95% CI]": f"{_fmt(overall['value'], scale)} [{_fmt(overall['ci_low'], scale)}, {_fmt(overall['ci_high'], scale)}]",
            "Huawei": _fmt(_endpoint(endpoints, metric, "Huawei")["value"], scale),
            "Azure": _fmt(_endpoint(endpoints, metric, "Azure")["value"], scale)})
    return result


def _table4(endpoints):
    result = []
    for vendor in ("Huawei", "Azure"):
        for category, metric, prefix in (("U", "p1_cost_ratio", "p1"),
                                          ("O", "p2_cost_ratio", "p2")):
            result.append({"Vendor": vendor, "Category": category,
                "Median aggregate": _fmt(_endpoint(endpoints, metric, vendor)["value"]),
                "Total paid ratio": _fmt(_endpoint(endpoints, metric, vendor, "total_cost")["value"]),
                "App/base P90": _fmt(_endpoint(endpoints, prefix + "_app_total_ratio_p90", vendor,
                                                 "total_cost")["value"])})
    return result


def _table5(descriptive):
    result = []
    for vendor in ("Huawei", "Azure"):
        for label, controller in (("nominal", "HPA_raw"), ("wrapped", "HPA_wrapped"),
                                  ("retuned", "known_delay_HPA")):
            row = _one(descriptive, section="known_delay_HPA", scope_level="vendor", scope=vendor,
                       case="PRIMARY_POOLED", controller=controller)
            result.append({"Vendor": vendor, "HPA version": label,
                "Compliant (%)": _fmt(row["compliance"], 100., 2),
                "Mean risk (%)": _fmt(row["risk_mean"], 100., 2),
                "Paid / nominal": _fmt(row["paid_ratio_to_nominal"]),
                "Paid / retuned": _fmt(row["paid_ratio_to_known_delay"])})
    return result


def _table6(summary):
    result = []
    for variant, label, scopes in (("risk_0.06", "6%", ("vendor/Huawei", "vendor/Azure", "overall")),
                                   ("risk_wrapper_actual", "Wrapped realized risk", ("overall",))):
        for category in ("U", "O"):
            for prefix in scopes:
                scope = f"{prefix}/class_{category}"
                value = lambda metric: _one(summary, variant=variant, scope=scope, metric=metric)["value"]
                result.append({"Static risk limit": label, "Category": category,
                    "Scope": {"vendor/Huawei": "Huawei", "vendor/Azure": "Azure", "overall": "Overall"}[prefix],
                    "Median cost ratio": _fmt(value("median_cost_ratio")),
                    "Feasible (%)": _fmt(value("feasible_fraction"), 100., 2),
                    "Total paid ratio": _fmt(value("total_paid_ratio"))})
    return result


def _table7(descriptive):
    result = []
    for case in CASES:
        item = {"Case": case}
        for category, bases in CLASSES.items():
            compliance = []
            for kind in ("raw", "wrapped"):
                rows = [_one(descriptive, section="eta_case_sensitivity", scope_level="overall",
                             scope="overall", case=case, contract_scope="all_selected", eta="0.1",
                             controller=base + "_" + kind) for base in bases]
                compliance.append(sum(float(row["compliance"]) for row in rows) / len(rows))
            item[f"{category} compliant B/W (%)"] = f"{_fmt(compliance[0], 100., 2)} / {_fmt(compliance[1], 100., 2)}"
            item[f"{category} total paid W/B"] = "INPUT_INCOMPLETE"
        result.append(item)
    return result


def _figure1(endpoints):
    rows = []
    for tolerance, variant in ((.1, "tolerance_0.001"), (.5, "tolerance_0.005"),
                               (1., "primary"), (2., "tolerance_0.02")):
        for series in ("exploration", "confirmation"):
            if series == "exploration":
                rows.append({"series": series, "tolerance_pp": tolerance, "estimate_pp": "INPUT_INCOMPLETE",
                             "ci_low_pp": "INPUT_INCOMPLETE", "ci_high_pp": "INPUT_INCOMPLETE"})
            else:
                row = _endpoint(endpoints, "p1_delta", variant=variant)
                rows.append({"series": series, "tolerance_pp": tolerance,
                             "estimate_pp": _fmt(row["value"], 100., 3),
                             "ci_low_pp": _fmt(row["ci_low"], 100., 3),
                             "ci_high_pp": _fmt(row["ci_high"], 100., 3)})
    return rows


def _extract_table(text, number):
    marker = re.search(rf"\*\*Table {number}\.[^\n]*\*\*", text)
    if not marker:
        raise ValueError(f"Missing Table {number} caption")
    lines = text[marker.end():].strip().splitlines()
    table = []
    for line in lines:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break
    if len(table) < 3:
        raise ValueError(f"Missing Markdown body for Table {number}")
    header = [cell.strip() for cell in table[0].strip("|").split("|")]
    return [dict(zip(header, (cell.strip() for cell in line.strip("|").split("|"))))
            for line in table[2:]]


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def reproduce(input_directory, output_directory, manuscript):
    source = Path(input_directory)
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise ValueError("Missing public aggregate CSVs: " + ", ".join(missing))
    data = {name: _read(source / name) for name in FILES}
    endpoints = data[FILES[0]]; descriptive = data[FILES[2]]; static = data[FILES[3]]
    tables = {3: _table3(endpoints), 4: _table4(endpoints), 5: _table5(descriptive),
              6: _table6(static), 7: _table7(descriptive)}
    destination = Path(output_directory); destination.mkdir(parents=True, exist_ok=True)
    comparisons = []
    manuscript_text = Path(manuscript).read_text(encoding="utf-8")
    for number, generated in tables.items():
        _write(destination / f"table{number}.csv", generated)
        expected = _extract_table(manuscript_text, number)
        for index, (actual, shown) in enumerate(zip(generated, expected), 1):
            for field, value in actual.items():
                observed = shown.get(field, "MISSING_COLUMN")
                status = "INPUT_INCOMPLETE" if value == "INPUT_INCOMPLETE" else "MATCH" if value == observed else "DIFF"
                comparisons.append(dict(artifact=f"Table {number}", row=index, field=field,
                                        generated=value, manuscript=observed, status=status))
        if len(generated) != len(expected):
            comparisons.append(dict(artifact=f"Table {number}", row="ALL", field="row_count",
                                    generated=len(generated), manuscript=len(expected), status="DIFF"))
    figure = _figure1(endpoints); _write(destination / "figure1.csv", figure)
    prose = re.search(r"At allowances of 0\.1, 0\.5, 1 and 2 pp above the target, the confirmatory P1 effects were ([0-9., and]+) pp", manuscript_text)
    displayed = re.findall(r"\d+\.\d+", prose.group(1)) if prose else []
    for row in figure:
        if row["series"] == "exploration":
            comparisons.append(dict(artifact="Figure 1", row=row["tolerance_pp"], field="exploration series",
                                    generated="INPUT_INCOMPLETE", manuscript="PLOTTED_PRIVATE_SOURCE", status="INPUT_INCOMPLETE"))
        else:
            index = (.1, .5, 1., 2.).index(row["tolerance_pp"])
            observed = displayed[index] if index < len(displayed) else "MISSING_PROSE_VALUE"
            comparisons.append(dict(artifact="Figure 1", row=row["tolerance_pp"], field="confirmation estimate_pp",
                                    generated=f"{float(row['estimate_pp']):.2f}", manuscript=observed,
                                    status="MATCH" if f"{float(row['estimate_pp']):.2f}" == observed else "DIFF"))
            for field in ("ci_low_pp", "ci_high_pp"):
                comparisons.append(dict(artifact="Figure 1", row=row["tolerance_pp"],
                                        field="confirmation " + field, generated=row[field],
                                        manuscript="PLOT_ONLY_NO_SOURCE_VALUE", status="INPUT_INCOMPLETE"))
    _write(destination / "comparison.csv", comparisons)
    statuses = {row["status"] for row in comparisons}
    overall = "DIFF" if "DIFF" in statuses else "INPUT_INCOMPLETE" if "INPUT_INCOMPLETE" in statuses else "PASS"
    return overall, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manuscript", required=True)
    args = parser.parse_args(argv)
    status, comparisons = reproduce(args.input_dir, args.output_dir, args.manuscript)
    counts = {name: sum(row["status"] == name for row in comparisons)
              for name in ("MATCH", "DIFF", "INPUT_INCOMPLETE")}
    print(f"{status}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

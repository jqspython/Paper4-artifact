# Paper4 code, selected workload identifiers and aggregate results

Software artifact: https://github.com/jqspython/Paper4-artifact
A repository or archive deposit is not performed by this exporter. Zenodo DOI: NOT_ASSIGNED.
The private Paper4 research repository is not made public by this snapshot.

## Scope and status

The code includes the discrete actuator, decision-loss feedback, PAC-h calibration,
four-base wrapper, analysis and data-preparation functions, synthetic property
tests, and the current small-scale system implementation. The complete v3 system
matrix (72 formal comparisons plus six fault runs) was CANCELLED_INCOMPLETE by
the project owner. Ten attempts yielded seven pre-clock failures and three
complete historical records; one was superseded, leaving two current records.
The release includes the offline-checked bootstrap retry candidate, not a newly
qualified workload sender. Its exported system configuration is explicitly
cancelled and omits private administrative receipt bindings; it is not the
historical frozen configuration and cannot authorize formal replay.

| Component/check | Validation status | Boundary |
|---|---|---|
| Early bounded 64-property snapshot | Historical synthetic checks | Later unit suites are separate; no workload comparison |
| Post-repair 300-second component check | Historical real container execution | 1 to 3 to 5 to 1; two +2 actions; three pending-addition blocked ticks; four downscale deletions; no spurious flag |
| Two current complete workload records | Historical real execution | Near-idle 99-request record and one out-of-contract cap demonstration; no controller-effect claim |
| Monitor/certificate demonstration | One historical cap trajectory | Same-tick cap detection and arithmetic agreement; not six-fault validation |
| Bootstrap health/warmup retry candidate | Offline tests only | No real-run capacity/fidelity requalification; extra warmup attempts are logged |
| Full 78-slot matrix | CANCELLED_INCOMPLETE | No paired direction, cost or service-SLO conclusion |
| Capacity and maximum-rate fidelity for this candidate | NOT_VALIDATED | Earlier static c=70 belongs to a historical binary, not this release's qualification |

In the cap trajectory, 14,044 of 22,230 planned requests had no recorded
connection acquisition or successful write before their deadline. The source
starts tasks on an exogenous schedule, but delivered service input did not
faithfully realize that schedule. Client/transport failures cannot be read as a
service SLO. One connection per Transport is a constraint, not an identified
unique cause of non-delivery; the retained same-slot planned gap exceeded the
request deadline. The original failures and earlier release tags are preserved.

The v1/v2 canonical command-line gates are intentionally unchanged. They refer to
original private Git freeze tags and protected source records that are absent
from this new repository. This snapshot directly supports pure-function and
synthetic engineering checks; it is NOT a self-contained rerun of the historical
canonical experiments or numerical manuscript reconstruction. Do not bypass the
gates and label a new execution as the original frozen study. An independently
documented public replay adapter and provenance package remain future work.

## Installation and checks (no scientific trace required)

Use Python 3.11 and install `requirements-gate2.txt` in an isolated environment.
Those exact versions record the local experiment environment; package availability
on another platform is a separate check. No dependencies or binaries are vendored.

```sh
python -m pip install -r requirements-gate2.txt
python -m pip install -e .
python -c "from pathlib import Path; Path('work').mkdir(exist_ok=True)"
python -B -m unittest discover -s tests -p 'test_*.py' -v
python -B v2_lite.py --stage plan
paper4-replay
paper4-wrapper --y-count 20 --n-start 1 --base-request 2
paper4-monitor
paper4-reproduce-synthetic
paper4-monitor-matrix --output work/e-b-monitor-summary.csv
paper4-regenerate-manuscript --input-dir results --output-dir work/reproduced --manuscript /path/to/submission.md
```

`paper4-replay` recomputes controller decisions from a supplied tick CSV or a
trace-free built-in fixture; it is not the cancelled live workload matrix.
`paper4-wrapper` exposes one wrapper decision, while `paper4-monitor` recomputes
observed contract loss and flags. The synthetic wrapper check also verifies its
conditional certificate arithmetic. `make reproduce-synthetic` invokes the same
bounded fixture as `python -B -X utf8 -m paper4_artifact_cli` and
`paper4-reproduce-synthetic`; this direct Python form is available when Make is
not installed. A PASS is an engineering check, not a trace result.

`paper4-monitor-matrix` runs 27 deterministic synthetic fault trajectories and
1,000 contract-valid trajectories. It tests monitor wiring only, without trace,
container or service traffic. `paper4-regenerate-manuscript` reads exactly the
seven public aggregate CSVs and emits Table 3--7 and Figure 1 data plus a cellwise
comparison. The current public inputs reproduce 167 source-text cells with zero
differences, but honestly return `INPUT_INCOMPLETE`: Table 7 lacks direct paid-sum
sufficient statistics and Figure 1 lacks the exploration series; plotted interval
endpoints are not numeric source text. Do not describe this release as complete
numerical manuscript regeneration. The unpublished manuscript is not bundled;
pass an authorized local source path to the comparator.

Tests construct their own synthetic arrays/SQLite fixtures and do not evaluate
scientific workload outcomes. The optional upstream PAC parity check skips when
the separately obtained official source cache is absent. Mathematical and causal
PAC checks still run. `tests/check_delayed_bound.py` is an additional synthetic
bound check. NumPy/Numba may write local caches, which must not be published.

The Go sender is Linux-specific (monotonic clock and affinity syscalls), uses only
the Go standard library, and needs a local Go toolchain. Its pure check is:

```sh
go test sysbench/replay.go sysbench/replay_test.go
```

Do not run Kubernetes/container commands to assess this release's unit checks.
The runtime adapter needs an isolated cluster, a locally built service image,
explicit private client credentials and independent fidelity/startup gates.
No cluster credentials, request schedules or request logs are included.

For an explicitly isolated engineering deployment example, build the pinned
service image and load it into the three-node kind definition:

```sh
docker build -t paper4-v3-http:feasibility sysbench
kind create cluster --name paper4-artifact --config sysbench/kind.yaml
kind load docker-image paper4-v3-http:feasibility --name paper4-artifact
kubectl apply -f sysbench/kind-service.yaml
```

This example only deploys the fixed-CPU service. It does not qualify capacity,
sender fidelity, the cancelled 78-slot comparison, or a real-service SLO claim.

The separately installable artifact image runs only the trace-free synthetic
controller/monitor check. It can be loaded into the same kind cluster as a Job:

```sh
docker build -f Dockerfile.artifact -t paper4-artifact:v0.8.0 .
kind load docker-image paper4-artifact:v0.8.0 --name paper4-artifact
kubectl apply -f sysbench/kind-artifact-check.yaml
kubectl logs job/paper4-artifact-synthetic
```

A `PASS_SYNTHETIC_ONLY` Job log is an artifact wiring check. It is not a
capacity qualification, workload replay, service-SLO result or replacement for
the cancelled matrix.

## Reused data, identifiers and aggregate results

No raw archives, demand arrays, prefix numerical statistics, trained models,
SQLite databases, replay request CSV/JSONL logs or unpublished manuscript/review
files are distributed. The following source-qualified identity/selection metadata
contain no measured workload values:

- `data/processed/gate2/selected-unit-ids.csv`: original 150 exploratory IDs.
- `data/processed/gate2/eligible-unit-pool.csv`: prefix-eligible ID frame only.
- `data/processed/v2-lite-confirm/selected-unit-ids.csv`: 57/80/80 confirmatory IDs.
- `results/v3-sys-fragment-ids.csv`: 12 design-frozen formal/reserve identities;
  the original `formal-proposed` role label is retained as provenance. Source/app,
  role, hash order, fixed window rule and selection seed are metadata, not outcomes.

Seven machine-readable result tables are also included. They contain aggregate
statistics and provenance fields, not per-application demand sequences or request
schedules: `v2-lite-confirm-endpoints.csv`, `v2-lite-confirm-risk.csv`,
`v2-lite-confirm-descriptive.csv`, `v1.1-static-reference-summary.csv`,
`v1.1-lambda-diagnostic.csv`, `v1.1-azure2021-qualification.csv` and
`v3-sys-closure.csv`. Each file is byte-bound to its named private result tag by
the exporter. The static-reference table is post-hoc; the Azure 2021 table records
an insufficient cohort and zero controller runs; the system closure is an
incomplete engineering record rather than a service-SLO comparison.

Huawei Private/Public 2023 are separate sources from the same cloud vendor.
Official release: https://github.com/sir-lab/data-release
2023 description: https://github.com/sir-lab/data-release/blob/main/README_data_release_2023.md
The release README states CC BY 4.0. Cite Joosen et al., *How Does It Function?
Characterizing Long-term Trends in Production Serverless Workloads*, SoCC 2023,
doi:10.1145/3620678.3624783. Azure Functions 2019 is obtained from
https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsDataset2019.md
and its official release links; its data are CC BY 4.0, separate from repository
code under MIT. Cite Shahrad et al., *Serverless in the Wild: Characterizing and
Optimizing the Serverless Workload at a Large Cloud Provider*, USENIX ATC 2020.
These are existing external data/paper identifiers, not this artifact's DOI.

The minute sequences are request counts per minute; Azure function counts are
summed by HashApp. Selection uses the first five days only (at least 1% nonzero
minutes, two active days and a positive first-three-day peak), then fixed ID hashes.
Private uses observed days 28–60; Public uses 0–25; Azure uses 1–14. Missing cells
are zero-imputed and an observed mask is retained locally. Fit/calibration/validation
end at minutes 4320/7200/11520. The preparer source documents official retrieval
and aggregation; downloading or processing is explicit and is not run at import.
Confirmation extraction still requires the original provenance gate described
above. Source attribution and dataset licensing remain separate from code MIT.

## Licenses and attribution

This project's own code and generated software documentation use MIT, copyright
Paper4 contributors; see `LICENSE`. The copied/adapted PAC-h margin work retains
Maxim Kozhinov's MIT notice in `THIRD_PARTY_NOTICES.md` and source attribution.
No claim is made to implement the complete upstream PAC-h experimental protocol
or the full BACC system. Trace-related identifiers retain the cited source
attribution; the software MIT license does not relicense third-party data.
Third-party dependencies retain their respective licenses and are installed
separately. See `SOURCE_MAP.csv` for the explicit release file inventory.

## Poisson capacity correction

The v1 value c=96 was measured with constant-rate arrivals and must not
be used for formal Poisson comparisons. The configuration records the
bounded recalibration status; a missing successful calibration blocks
formal execution. D3 SLO/deadline and D1-D10 semantics are unchanged.
Calibration and a constant-load actuator component check are engineering
checks, separate from both the six earlier shakedowns and study endpoints.

The second-round T1 sender uses 256 independent single-connection transports,
preconnected before measurement and assigned by planned request ID modulo 256.
The original synthetic scaling check exposed connections pinned to the initial Pod.
The same T1 scheme now closes idle connections only when Ready membership changes;
planned request IDs, deadlines, Service routing and all D1-D10 semantics are retained.
The final twelfth calibration checked the historical Ready-refresh sender at unchanged c=70,
before the offline-only bootstrap retry candidate in this release,
and one synthetic scaling regression exercised redistribution and actuator paths.
Both rounds contain 17 calibration attempts in total; earlier failures are retained locally.
These checks are not formal controller comparisons or proof of production SLO performance.
The 78-slot comparison was cancelled incomplete; startup repairs remain offline-only.
The host guard retains update-reboot checks and records narrowly reviewed delayed cleanup
without editing the OS queue. Unknown operations or changed identities still block;
Windows Update Agent and a measured update-pause window are also checked before launch.

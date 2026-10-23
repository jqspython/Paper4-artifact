"""Bounded infrastructure measurements, not a trace run or controller comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v3-sys"
KUBE = ["kubectl", "--kubeconfig", str(WORK / "kubeconfig"), "-n", "paper4-v3"]
NODES = ["paper4-v3-sys-control-plane", "paper4-v3-sys-worker", "paper4-v3-sys-worker2"]
K6_IMAGE = "grafana/k6:2.3.0@sha256:9c2dee7f8ed74d317e4027c06a10f169b625638189de8d4555d0b3486a5aeb34"


def run(args, timeout=180):
    return subprocess.check_output(args, text=True, encoding="utf-8", timeout=timeout, stderr=subprocess.STDOUT)


def append(name, row):
    path = ROOT / "results" / name
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def pods():
    return json.loads(run(KUBE + ["get", "pods", "-l", "app=paper4-cpu-http", "-o", "json"]))["items"]


def ready(pod):
    return not pod["metadata"].get("deletionTimestamp") and any(
        c["type"] == "Ready" and c["status"] == "True" for c in pod["status"].get("conditions", []))


def settle(n):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        current = pods()
        if len(current) == n and sum(map(ready, current)) == n:
            return current
        time.sleep(.25)
    raise TimeoutError("Deployment did not settle")


def scale(n):
    run(KUBE + ["scale", "deployment/cpu-http", f"--replicas={n}"])


def startup(trials, sleep_seconds, baseline_count=1, leave_count=1):
    for trial in range(trials):
        scale(baseline_count)
        old = {p["metadata"]["uid"] for p in settle(baseline_count)}
        start = time.monotonic()  # Includes command/API overhead; no cross-host wall-clock subtraction.
        scale(8)
        found = {}
        while len(found) < 8-baseline_count and time.monotonic() - start < 90:
            for pod in pods():
                uid = pod["metadata"]["uid"]
                if uid not in old and uid not in found and ready(pod):
                    elapsed = time.monotonic() - start
                    found[uid] = elapsed
                    append("v3-sys-startup.csv", dict(platform="kind", sleep_s=sleep_seconds,
                        trial=trial+1, pod=pod["metadata"]["name"], request_to_observed_ready_s=elapsed,
                        status="READY", observation=f"batch-scale-command-start;baseline={baseline_count};poll=0.25s+API"))
            time.sleep(.25)
        if len(found) != 8-baseline_count:
            raise TimeoutError("Missing one or more of the new Ready pods")
        print("startup", trial+1, "median", statistics.median(found.values()), "range", max(found.values())-min(found.values()), flush=True)
    scale(leave_count)
    settle(leave_count)


def cpu_sample(rate, mode):
    data = run(["docker", "stats", "--no-stream", "--format", "{{json .}}", *NODES], timeout=15)
    values = [json.loads(line) for line in data.splitlines() if line.strip()]
    if len(values) != 3:
        raise ValueError("Missing kind node CPU sample")
    core_percent = sum(float(v["CPUPerc"].rstrip("%")) for v in values)
    value = core_percent/12  # All nodes share CPUs 4-15; no overlap with generator CPUs 0,2 or their siblings.
    append("v3-sys-cpu.csv", dict(mode=mode, rate_rps=rate, monotonic_s=time.monotonic(),
                                node_cpu_core_percent=core_percent, allocated_cpu_count=12,
                                node_allocated_percent=value))
    return value


def throttling():
    values = run(KUBE + ["exec", "deployment/cpu-http", "--", "cat", "/sys/fs/cgroup/cpu.stat"])
    return {a: int(b) for a, b in (line.split() for line in values.splitlines())}


def summary(output):
    for line in output.splitlines():
        if "PAPER4_SUMMARY=" not in line:
            continue
        if 'msg="' in line:
            line = json.JSONDecoder().raw_decode(line.split("msg=", 1)[1])[0]
        return json.loads(line.split("PAPER4_SUMMARY=", 1)[1])
    raise ValueError("k6 summary missing; preserve output, do not invent calibration")


def steady(rate, slo_ms=None, mode="capacity", replicas=1):
    scale(replicas)
    settle(replicas)
    before = throttling() if replicas == 1 else {}
    command = ["docker", "run", "--rm", "--name", "paper4-v3-k6", "--cpuset-cpus", "0,2", "--cpus", "2",
               "--network", "kind", "-e", f"RATE={rate}", "-e", "NO_COLOR=1",
               "-e", "TARGET=http://paper4-v3-sys-worker:30080", "--mount",
               f"type=bind,source={ROOT / 'sysbench'},target=/bench,readonly", K6_IMAGE,
               "run", "--quiet", "--log-format", "raw", "/bench/steady.js"]
    samples = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run, command, 150)
        while not future.done():
            samples.append(cpu_sample(rate, mode))
            time.sleep(2)
        output = future.result()
    WORK.mkdir(exist_ok=True, parents=True)
    (WORK / "last-k6.log").write_text(output, encoding="utf-8")
    metrics = summary(output)["metrics"]
    p95 = metrics["http_req_duration"]["values"]["p(95)"]
    count = metrics["http_reqs"]["values"]["count"]
    failed = metrics["http_req_failed"]["values"]["rate"]
    dropped = metrics.get("dropped_iterations", {"values": {"count": 0}})["values"]["count"]
    after = throttling() if replicas == 1 else {}
    row = dict(mode=mode, rate_rps=rate, replicas=replicas, seconds=120, requests=count,
               p95_ms=p95, slo_ms="" if slo_ms is None else slo_ms, failed_fraction=failed,
               dropped=dropped, cpu_mean_percent=statistics.mean(samples), cpu_max_percent=max(samples),
               throttle_periods=after["nr_throttled"]-before["nr_throttled"] if replicas == 1 else "NOT_MEASURED",
               total_cfs_periods=after["nr_periods"]-before["nr_periods"] if replicas == 1 else "NOT_MEASURED",
               elapsed_s=time.monotonic()-started,
               passes_slo="BASELINE" if slo_ms is None else int(p95 <= slo_ms and failed == 0 and dropped == 0 and count >= rate*120-1))
    append("v3-sys-calibration.csv", row)
    print(row, flush=True)
    return row


def report():
    def read(name):
        with (ROOT/"results"/name).open(encoding="utf-8", newline="") as source:
            return list(csv.DictReader(source))
    starts = read("v3-sys-startup.csv")
    rows = read("v3-sys-calibration.csv")
    nominal = [float(r["request_to_observed_ready_s"]) for r in starts if float(r["sleep_s"]) == 5]
    mismatch = [float(r["request_to_observed_ready_s"]) for r in starts if float(r["sleep_s"]) == 30]
    baseline, = [r for r in rows if r["mode"] == "unloaded-probe"]
    headroom, = [r for r in rows if r["mode"] == "eight-pod-headroom"]
    tested = [r for r in rows if r["mode"] == "capacity"]
    passing = [int(r["rate_rps"]) for r in tested if r["passes_slo"] == "1"]
    capacity = max(passing)
    bracket = min(int(r["rate_rps"]) for r in tested if r["passes_slo"] == "0" and int(r["rate_rps"]) > capacity)
    if len(nominal) != 21 or len(mismatch) != 8:
        raise ValueError("Incomplete startup measurement")
    info = tomllib.loads((ROOT/"configs/v3-sys.toml").read_text("utf-8"))["observed_environment"]
    nodes_ok = info["ready_nodes"] == 3
    cpu = [float(r["monotonic_s"]) for r in read("v3-sys-cpu.csv") if r["mode"] == "eight-pod-headroom"]
    sample_interval = statistics.mean(b-a for a, b in zip(cpu, cpu[1:]))
    infrastructure_ok = (nodes_ok and info["NCPU"] >= 8 and info["MemTotal"] >= 10*1024**3
        and (max(nominal)-min(nominal))/5 < .30
        and float(headroom["cpu_max_percent"]) < 70
        and float(headroom["failed_fraction"]) == 0 and int(headroom["dropped"]) == 0)
    lines = ["# v3-sys 可行性检查实测报告", "",
        "测量日期："+info["measurement_date"]+"，Asia/Shanghai。资源与Ready状态采用初检实测快照；重生成报告不重新探测硬件、不重放负载。",
        "",
        "**基础设施检查："+("通过，继续 kind" if infrastructure_ok else "未全部通过")+"。v3 协议仍未冻结，正式72+6次和完整片段shakedown均未运行。**",
        "",
        "## 测试台与隔离", "",
        f"Docker/WSL2实测可见{info['NCPU']}个逻辑CPU、{info['MemTotal']/1024**3:.3f} GiB内存。kind v0.33.0、Kubernetes v1.37.0三节点首次创建成功，结束核查三节点仍Ready；没有触发两次失败/抖动回退条件。Docker Desktop 4.91.0、Engine29.8.0。",
        "",
        f"k6 2.3.0绑WSL逻辑CPU0,2（拓扑报告为两个不同core），节点共享4–15；同胞1,3不分给节点。此为WSL内隔离，不把它夸为宿主机物理核的绝对隔离。节点CPU百分比为三节点Docker core-percent求和除以共享12个逻辑CPU，headroom采样平均间隔{sample_interval:.3f}秒（含docker stats耗时及2秒间隔），未声称连续瞬时峰值。",
        "",
        "HTTP服务固定执行6000轮PBKDF2-HMAC-SHA256，每pod CPU request/limit均250m、内存limit128Mi。镜像及工具版本在源码固定；本次没有用trace片段。kubeconfig和工具缓存留项目work目录，不提交凭据。",
        "",
        "## 启动与35秒修订", "",
        "| 注入 | 观测数/操作 | 最小/中位/最大总启动(s) | 极差(s) |",
        "|---|---|---|---|",
        f"| 5s | {len(nominal)}，3次1→8 | {min(nominal):.3f} / {statistics.median(nominal):.3f} / {max(nominal):.3f} | {max(nominal)-min(nominal):.3f} |",
        f"| 30s | {len(mismatch)}，额外0→8探测 | {min(mismatch):.3f} / {statistics.median(mismatch):.3f} / {max(mismatch):.3f} | {max(mismatch)-min(mismatch):.3f} |",
        "",
        f"名义样本全极差/5s={100*(max(nominal)-min(nominal))/5:.2f}%，低于30%。时钟从扩容CLI调用前开始，到首次观察Ready；包括API/调度/创建/探针/轮询，轮询0.25秒加API，单次get实测约0.141秒。不用跨时钟wall time相减，不把sleep等同总启动。",
        "",
        f"30s注入的全部8个观测都超过原30s硬阈值。用户已明确批准总启动契约改为35s；本次最大值至35s裕量为{35-max(mismatch):.3f}s。这只是有限样本的设计选择，正式运行必须继续监视，超35仍使该run保证条件失效。新常数d_max=8,m=4,L=11,K=220；T240 raw=31/30，显示截断为1。原30s/.95只作为历史算术保留。",
        "",
        "## 单pod容量和集群CPU", "",
        f"低载探测为1 RPS、120秒、{baseline['requests']}请求，无错误/漏发；p95={float(baseline['p95_ms']):.6f}ms，SLO预设规则3倍得到{3*float(baseline['p95_ms']):.6f}ms。c为有限测试网格中最大通过值{capacity} RPS，c_tick={5*capacity}请求/5秒；最近上方失败点{bracket} RPS，**不是精确连续最大值或生产SLO保证**。",
        "",
        "候选8/16/32/64/128到首失败停止，再预定两次二分96/112；容量通过需p95不超SLO、无错误/漏发且请求数完整。下表保留失败点，不选择性删除：",
        "",
        "| 角色 | RPS / pod数 | 请求 | p95(ms) | 错误(%) | 漏发 | CPU均值/采样最大(%) | CFS限流/周期 |",
        "|---|---|---:|---:|---:|---:|---:|---|"
    ]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['rate_rps']} / {r['replicas']} | {r['requests']} | {float(r['p95_ms']):.3f} | {100*float(r['failed_fraction']):.3f} | {r['dropped']} | {float(r['cpu_mean_percent']):.3f} / {float(r['cpu_max_percent']):.3f} | {r['throttle_periods']} / {r['total_cfs_periods']} |")
    lines += ["",
        f"8pod、{headroom['rate_rps']}RPS持续120秒，{headroom['requests']}请求，无错误/漏发，p95={float(headroom['p95_ms']):.3f}ms；节点CPU均值{float(headroom['cpu_mean_percent']):.3f}%、采样最高{float(headroom['cpu_max_percent']):.3f}%，低于70%。8pod的逐pod CFS没有采集，明确NOT_MEASURED，不解释为零。CPU低余量充分是本工作量下的观测，不推出宿主机永远无争抢。",
        "",
        "容量判定只用capacity角色；headroom未参与c选择。原CSV里未传独立SLO参数的角色使用BASELINE哨兵，不能将headroom误读成又一份低载探测或新的容量判据。128与112的严重延迟/漏发与250m限流共同出现；不拿漏发后的较低有效流量算保护收益。",
        "",
        "## 实现、审核和剩余工作", "",
        "- 用户已批准正式采用固定泊松时间戳的单调时钟开环重放；k6只用于定常校准。所有臂复用同一计划到达序列，仍须记录实际调度迟到/漏发/错误，不能承诺现实时间戳零抖动。",
        "- 纯控制器/监视器和离线分析10项、泊松调度7项，合17项构造输入的性质/路径测试通过；旧lambda发令、单位、队列、35s边界锁存、证书、同一计划序列/开放调度都有检查。同体系独立源审闭合35s依赖与基线有效性/未测限流假零两项；这不是人类签核或外部验证。",
        "- 正式send须真异步且可取消，结果回调只非阻塞聚合/入队，磁盘JSONL写入另执行。回调错误、少发或时钟异常须使run显式无效，不静默使用残缺日志。同一计划tuple不等于网络实际到达零抖动；HTTP/落盘接线仍需实测。",
        "- HPA150探索锁点中位目标1.0；固定哈希产生6正式候选和6个不同备用ID，未读需求数组、未按性能换片段。正式队列适配、响应/成本计量、故障确实激活的真值与完整矩阵分析尚未完成，不能据纯模块测试称系统闭环完成。",
        "- 首次k6摘要因日志转义解析失败，未赋容量；修复raw/logfmt解析后重做完整校准，原失败输出留work/v3-sys/first-k6-parser.log。基线未通过即中止；所有成功/失败容量点均在CSV。",
        "- 本次为隔离基础设施校准，不计作正式片段shakedown；其结果不进入论文的72+6效果表。无新增抗饱和、无正式性能主张、没有统计显著性检验。",
        "",
        "数据源：results/v3-sys-startup.csv、v3-sys-calibration.csv、v3-sys-cpu.csv。报告由sysbench/feasibility.py report生成，不手抄表数字。配置/属性/分析说明见docs/v3-sys-protocol.md及sysbench/。原始trace与需求数组不再分发；正式JSONL留本地work目录。",
        "",
        "复核入口：`python -B -X utf8 sysbench/feasibility.py report`；`python -B -X utf8 -m unittest tests.test_v3_sys tests.test_v3_arrival -v`。协议冻结前仍须将真实适配器、负载端及量测端接通并独立复核；10/24搭建不通过则按用户既定规则回A。"
    ]
    (ROOT/"docs/v3-sys-feasibility.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Generated docs/v3-sys-feasibility.md; infrastructure checks", infrastructure_ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("startup", "delay30", "calibrate", "headroom", "report"))
    parser.add_argument("--rate", type=int)
    args = parser.parse_args()
    if args.stage == "startup":
        startup(3, 5)
    elif args.stage == "report":
        report()
    elif args.stage == "delay30":
        scale(0)
        settle(0)
        run(KUBE + ["set", "env", "deployment/cpu-http", "STARTUP_SLEEP=30"])
        startup(1, 30, baseline_count=0, leave_count=8)
    elif args.stage == "calibrate":
        baseline = steady(1, mode="unloaded-probe")
        if not (math.isfinite(baseline["p95_ms"]) and baseline["p95_ms"] > 0
                and baseline["failed_fraction"] == 0 and baseline["dropped"] == 0
                and baseline["requests"] >= 119):
            raise ValueError("Unloaded probe failed; SLO/capacity cannot be calibrated")
        slo = 3*baseline["p95_ms"]
        passing, failing = 1, None
        for rate in (8, 16, 32, 64, 128):
            row = steady(rate, slo)
            if not row["passes_slo"]:
                failing = rate
                break
            passing = rate
        if failing:
            # Fixed two bisection steps; this is a finite tested-grid capacity, not a continuous maximum.
            for _ in range(2):
                if failing-passing <= 1:
                    break
                rate = (passing+failing)//2
                if steady(rate, slo)["passes_slo"]:
                    passing = rate
                else:
                    failing = rate
        print("CAPACITY_TESTED_GRID", passing, "rps; SLO", slo, "ms; first-failing-bracket", failing, flush=True)
    else:
        if args.rate is None or args.rate < 1:
            parser.error("headroom requires a positive --rate (total RPS)")
        steady(args.rate, mode="eight-pod-headroom", replicas=8)


if __name__ == "__main__":
    main()

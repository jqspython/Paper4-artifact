"""Regenerate D9/D10 tables from retained local requests and calibration records."""
import csv
import json
import math
from pathlib import Path
import statistics

from sysbench.analyze import analyze
from sysbench.feasibility import summary
from sysbench.gates import independent_requests

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/"work/v3-sys"


def k6_values(label):
    with (WORK/(label+"-metrics.csv")).open(encoding="utf-8") as source:
        values=sorted(float(r["metric_value"]) for r in csv.DictReader(source) if r["metric_name"]=="planned_latency_ms")
    metrics=summary((WORK/(label+".log")).read_text("utf-8"))["metrics"]
    errors=metrics["http_req_failed"]["values"]["rate"]
    dropped=metrics.get("dropped_iterations",{}).get("values",{}).get("count",0)
    custom_errors=metrics.get("crosscheck_failures",{}).get("values",{}).get("count",0)
    return dict(count=len(values),p95=values[math.ceil(.95*len(values))-1],minimum=min(values),
                eligible=min(values)>=0 and errors==0 and dropped==0 and custom_errors==0)


def report():
    rows=[]
    for session,label in ((1,"s01-fidelity768"),(2,"s02-cross96-go")):
        primitive=independent_requests(WORK/(label+"-requests.csv"))
        messages=[json.loads(line) for line in (WORK/(label+"-events.jsonl")).read_text("utf-8").splitlines()]
        online,=[r for r in messages if r["type"]=="complete"]
        for key in ("planned","missing","retried_requests","failed","late_p99_ms","latency_p95_ms"):
            other={"missing":"missing_write"}.get(key,key)
            if abs(primitive[key]-online[other])>1e-9: raise ValueError("Go/CSV gate summary mismatch")
        rows.append(dict(session=session,check="sender768_initial" if session==1 else "go96",
            status="SUPERSEDED_TRANSPORT" if session==1 else "PASS_CROSSCHECK_COMPONENT",
            count=primitive["planned"],p99_ms=primitive["late_p99_ms"],p95_ms=primitive["latency_p95_ms"],
            cpu_bound_percent=online["sender_bound_cpu_fraction"]*100,failed=primitive["failed"],
            missing=primitive["missing"],retried=primitive["retried_requests"],maximum_startup_seconds="",relative_difference=""))
    go=rows[1]
    for session,label in ((3,"s03-cross96-k6"),(4,"s04-cross96-k6")):
        data=k6_values(label)
        difference=abs(go["p95_ms"]/data["p95"]-1)
        rows.append(dict(session=session,check="k6_crosscheck",status="INVALID_CLOCK_ALIGNMENT" if not data["eligible"]
            else "PASS" if difference<=.10 else "FAIL",count=data["count"],p99_ms="",p95_ms=data["p95"],
            cpu_bound_percent="",failed=0,missing="",retried="",maximum_startup_seconds="",relative_difference=difference))
    startup=json.loads((WORK/"s05-loaded-startup-30.json").read_text("utf-8"))
    selected=25 if startup["maximum"]>33.5 else 30
    measured=startup if selected==30 else json.loads((WORK/"s05-loaded-startup-25.json").read_text("utf-8"))
    load=summary((WORK/"s05-loaded-k6.log").read_text("utf-8"))["metrics"]
    loaded_ok=(len(measured["starts"])==8 and measured["maximum"]<=35
        and load["http_req_failed"]["values"]["rate"]==0
        and load.get("dropped_iterations",{}).get("values",{}).get("count",0)==0)
    rows.append(dict(session=5,check="loaded_startup",status="PASS" if loaded_ok else "FAIL",
        count=load["http_reqs"]["values"]["count"],p99_ms="",p95_ms=load["http_req_duration"]["values"]["p(95)"],
        cpu_bound_percent="",failed=load["http_req_failed"]["values"]["rate"],missing="",retried="",
        maximum_startup_seconds=startup["maximum"],relative_difference=""))
    actual=analyze(WORK/"s06-final-control.jsonl",expected_ticks=60)
    raw=[json.loads(line) for line in (WORK/"s06-final-control.jsonl").read_text("utf-8").splitlines()]
    completed,=[r for r in raw if r["type"]=="complete"]
    primitives=independent_requests(WORK/"s06-final-control-requests.csv")
    fidelity_ok=(primitives["late_p99_ms"]<=min(5,18.924*.1) and completed["sender_bound_cpu_fraction"]<.30
                 and primitives["missing"]==0 and primitives["retried_requests"]==0)
    rows.append(dict(session=6,check="final_sender_and_control",status="PASS" if fidelity_ok else "FAIL",
        count=primitives["planned"],p99_ms=primitives["late_p99_ms"],p95_ms=primitives["latency_p95_ms"],
        cpu_bound_percent=completed["sender_bound_cpu_fraction"]*100,failed=primitives["failed"],missing=primitives["missing"],
        retried=primitives["retried_requests"],maximum_startup_seconds="",relative_difference=""))
    target=ROOT/"results/v3-sys-gates.csv"
    with target.open("w",newline="",encoding="utf-8") as out:
        writer=csv.DictWriter(out,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    text=["# v3-sys D1–D10 工程门实测", "", "日期：2026-10-23。表由 `python -m sysbench.report_gates` 从本地原始请求、时钟与Pod生命周期记录生成。正式72+6仍NOT_RUN；不是科研端点通过。", "",
        "| 次数 | 检查 | 状态 | 数量 | p99迟到(ms) | p95(ms) | 发送器绑定核CPU(%) | 请求失败 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(value): return f"{value:.6f}" if isinstance(value,(int,float)) else str(value)
        text.append("|"+"|".join(str(row[k]) if k in ("session","check","status","count","failed") else fmt(row[k])
            for k in ("session","check","status","count","p99_ms","p95_ms","cpu_bound_percent","failed"))+"|")
    text += ["",f"D10单pod同口径交叉差={rows[3]['relative_difference']:.4%}（阈值10%）。Go与k6都从场景计划时刻计至完成；HTTP纯传输耗时仅辅报。第3次有负延迟，按时钟原点不一致作废保留，未隐藏或拿来达标。", "",
        f"D9：8个既有Ready pod承载400RPS的连续120秒流量，同时创建8个新pod；临时最多16pod，仅用于带载启动，不是R=8正式臂。30秒注入的总启动中位{statistics.median(startup['starts']):.6f}s，最大{startup['maximum']:.6f}s；严格大于33.5s条件{'触发' if selected==25 else '未触发'}，因此正式注入采用5/15/{selected}秒，未做无必要替换测量。", "",
        "客户端在第1次后加入Linux TCP_QUICKACK，不修改服务CPU工作或容量，并重新做最终保真度。96RPS交叉已一致，但最终768RPS仍有高响应尾部（见表）和60/60个SLO窗口违约；TCP确认、负载分配和CPU限流的相对作用未被包级/逐Pod测量分离，不能宣称根因已解决。高响应尾部不被发送门通过掩盖。", "",
        "第6次为固定泊松768RPS、60tick、初始8Ready的工程探针，复用第1次完整到达计划；不同于正式240tick/初始1Pod。它实测发送、实际边界、Ready采样、lambda记录、派发接口、反馈、逐tick证据与离线分析。实际命令恒8，不能单凭此轨迹区分用旧/新lambda派发，也未实测正式从1Pod的完整+2扩容/在途禁缩/正常缩容路径；这些路径由性质检查、源审及后续正式运行分别核对。该负载在部分tick计划到达超过8pod包络，因此监视器锁存是独立真越界，不能当误报或用于声称5%风险控制。", "",
        "离线第6次检查（primitive事件/原计划/逐请求CSV；脚本生成）：", "", "| 项目 | 复算值 |", "|---|---|"]
    brief={"实际容量违约率":actual["capacity_violation_rate"],"SLO窗口违约率":actual["latency"]["slo_violation_rate"],
        "付费副本秒":actual["paid_replica_seconds"],"Ready副本秒":actual["ready_replica_seconds"],
        "证书最大算术误差":actual["max_certificate_error"],"4–6s实际tick比例":actual["loop_in_range_fraction"],
        "原计划与逐请求内容校验":actual["schedule_hash_content_verification"],
        "独立包络越界存在":actual["monitor"]["fault_exercised"],"误报":actual["monitor"]["false_positive"],
        "检测实际tick差":actual["monitor"]["detection_delay_ticks"],
        "证书适用性":actual["certificate_applicability"],"正式新扩容离散lag":actual["discrete_lag_status"]}
    text += [f"| {key} | {value} |" for key,value in brief.items()]
    text += ["",
        "6/6试运行额度已使用；第3次无效也计入。所有本地原始日志保留在work/v3-sys，不公开需求、请求时间序列或凭据。未完成正式监视器6/6、72次误报、方向和成本端点，均不得因本门通过宣称通过。", "",
        "使用：CCFA实验设计对齐D1–D10及证据原点；Nature统计规范将片段与重复分开、不对工程探针做显著性推断；scientific-critical-thinking区分源级性质、运行实测和有限范围结论；Ponytail复用标准库与既有控制核心。独立审阅为同一Codex体系，不是人类或外部复核。", "",
        "人类证明/核心结论负责人仍用户本人，尚无完成签核；文献/引用10月25日待人工核验。", "",
        "接口依据：[Go httptrace](https://pkg.go.dev/net/http/httptrace)、[k6 execution](https://grafana.com/docs/k6/latest/javascript-api/k6-execution/)、[Linux TCP options](https://man7.org/linux/man-pages/man7/tcp.7.html)。"]
    (ROOT/"docs/v3-sys-gates.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(json.dumps(dict(sender=fidelity_ok,crosscheck=rows[3]["status"],loaded=loaded_ok,
        selected_injection=selected,control=actual),ensure_ascii=False,allow_nan=False))


if __name__=="__main__": report()

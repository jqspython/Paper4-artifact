"""Render the bounded calibration and component evidence from retained CSVs."""
import csv
import argparse
import json
from pathlib import Path

from sysbench.calibrate import ROOT, WORK, LEDGER


def component_distribution(records, requests):
    """Descriptive check at the largest concurrently Ready set; no endpoint or extra load."""
    origin,=[r['start_mono_ns']/1e9 for r in records if r['type']=='start']
    events=[r for r in records if r['type']=='pod_event']
    readies={r['pod']:r['at'] for r in events if r['event_type']=='ready'}
    deleted={r['pod']:r['at'] for r in events if r['event_type']=='delete'}
    tagged=[]
    for r in requests:
        when=origin+int(r['offset_ns'])/1e9
        ready=tuple(sorted(p for p,t in readies.items() if t<=when<deleted.get(p,float('inf'))))
        tagged.append((ready,r.get('pod','')))
    largest=max((len(ready) for ready,_ in tagged),default=0)
    selected=[(ready,pod) for ready,pod in tagged if len(ready)==largest]
    counts={p:0 for ready,_ in selected for p in ready}
    missing=0
    for ready,pod in selected:
        if pod in ready: counts[pod]+=1
        else: missing+=1
    return dict(maximum_ready=largest,planned_requests=len(selected),pod_counts=counts,
                missing_or_not_ready=missing)


def report():
    with LEDGER.open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    decision_path=WORK/"capacity-v2-decision.json"
    decision=json.loads(decision_path.read_text("utf-8")) if decision_path.exists() else dict(status="RUNNING",selected_capacity=None)
    lines=["# v3-sys 正式泊松过程下的容量重校准", "", "日期：2026-10-24，Asia/Shanghai。", "",
        "旧c=96来自恒定速率有限网格，用户禁止将其用于正式78槽。旧v3-sys-design-v1不移动；此次全部短运行与已耗尽的6次shakedown分开登记，使用固定合成到达，不读取正式片段需求。", "",
        "规则：4Ready pod，每pod40/50/60/70/80RPS各180秒；候选及上一档各另重复3次。36个非空5秒窗下≤2%意味着零违约窗；错误按∞且不得删分母。每pod请求最大/均值≤1.3，零失败。若最低候选40，30只作为已声明的相邻参考，不可选为c。", "",
        "用户已在运行前批准合并以维持12次上限：首点含均衡检查；末次8pod300秒，前180秒要求SLO窗口≤5%（最多1/36）且零失败，完整300秒检查D10的p99≤1.8924ms、CPU<30%。若退档无额度则停止；不添加第13次。", "",
        "初筛种子20261024，三重复2026102401/02/03，末次2026102499。HTTP工作始终6000次PBKDF2且250m；只添加返回pod身份头以审分发，源代码/新镜像一致用于本轮所有点及未来臂。", "",
        "| 次数 | 阶段 | pod数×每podRPS | 连接复用关闭 | 前缀坏窗/窗数 | 前缀/全程失败 | pod最大/均值 | p99迟到ms | 绑定核平均CPU% | SLO / 保真度 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        measured=r["status"]=="MEASURED"
        ratio=f"{float(r['backend_max_over_mean']):.4f}" if measured else r["status"]
        cpu=f"{float(r['sender_cpu_fraction'])*100:.4f}" if measured else ""
        lines.append("|"+"|".join((r["number"],r["purpose"],r["pods"]+"×"+r["per_pod_rps"],r["fresh_connections"],
            r["slo_bad_windows"]+"/"+r["slo_windows"],r["prefix_failed"]+"/"+r["failed"],ratio,r["late_p99_ms"],cpu,
            r["slo_pass"]+" / "+r["fidelity_pass"]))+"|")
    lines += ["",f"实际已用校准短运行：{len(rows)}/12。决策：**{decision['status']}**；选定c：**{decision.get('selected_capacity')}**。", "",
        "每条原始计划、实际写入/完成/错误与响应pod保存在本地work/v3-sys/cal-v2-*；不公开请求时间序列。CSV的failed是全程失败，prefix_failed是SLO判定前缀失败；末次二者不可混称。grid/repeat的fidelity列仅描述，不用于替代其容量SLO规则；D10只在末次裁定。"]
    for r in rows:
        if r["status"]=="INFRASTRUCTURE_INVALID":
            records=[json.loads(s) for s in (WORK/(r["label"]+"-events.jsonl")).read_text("utf-8").splitlines()]
            started,=[e for e in records if e["type"]=="start"]
            driver,=[e for e in records if e["type"]=="calibration_driver"]
            boundaries=[e for e in records if e["type"]=="boundary"]
            elapsed=driver["finished_mono"]-started["start_mono_ns"]/1e9
            error_log=(WORK/(r["label"]+"-host.log")).read_text("utf-8")
            reason="Ready Pod became NotReady" if "Ready Pod became NotReady" in error_log else "见原始错误记录"
            lines += ["",f"第{r['number']}次（每pod{r['per_pod_rps']}RPS）为中止点：日志原因 `{reason}`；只有start、{len(boundaries)}个边界与driver记录，计划起点至driver结束{elapsed:.6f}s，没有完整180秒结果或逐请求CSV。窗口、失败数、发送p99均不可补造。Kubernetes具体状态快照未留存，不能进一步确定Ready回退根因，也不改称已证实的实现缺陷。"]
    if decision.get("next_stage")=="FALLBACK_A":
        lines += ["",f"按固定流程，重做缺失的完整初筛点后还需六次重复与一次最终检查，最低总数为{decision['minimum_runs_if_retry_then_confirm']}，超过12。故不追加运行、不建v3-sys-design-v2、正式78槽不启动；系统比较退回未来工作。未运行候选重复、8pod验证和新最大速率D10。",
            "", "失败的正确范围是本轮测试台/发送链组合未获得资格容量；不能据此宣称fresh分发下40RPS已被检验、物理容量低于30RPS、SLO在所有速率上不可能，或CFS是唯一根因。写回调缺失只表明未观察到成功完成写出，不足以判断请求完全未上网。"]
    with (ROOT/"results/v3-sys-component.csv").open(encoding="utf-8") as stream: component=list(csv.DictReader(stream))
    lines += ["", "## 实时扩缩容组件检查（非端点）", "",
        "唯一一次300秒恒定1RPS，从1个Ready出发，脚本化基础建议用于隔离执行器路径；不是HPA效果或片段回放。组件c=40仅为安全计数单位，不是选定的物理容量。原始运行后用同一events/requests加强离线核accepted动作、Ready/删除和300计划请求，没有新增负载。", "",
        "| 状态 | +2动作 | 在途禁缩tick | 缩容动作 | 五tick稳定 | 取消在途 | 监视标志 | 实际动作/请求 |", "|---|---|---|---|---|---|---|---|"]
    for r in component:
        lines.append("|"+"|".join(str(r.get(k,"")) for k in ("status","plus_two_actions","inflight_downscale_blocked_ticks",
            "shrink_actions","five_tick_stability","cancelled_inflight","monitor_flags"))+"|"+
            r.get("accepted_actions_match","")+" / "+r.get("request_execution","")+"|")
    lines += ["", "公开仓库全部历史另查：原artifact-v0.1仅一个提交/70个文件，逐blob白名单、ID schema与敏感字面量检查通过，无不可达对象；不是仅检查最新树。新版公开时再次核整个历史，数据/凭据不入库。", "",
        "技能实效：CCFA实验设计固定整数窗口、预算和选择函数；Nature统计将重复与独立片段分开，不加显著性推断；scientific-critical-thinking区分分发失衡、容量拐点和未隔离的因果解释；Ponytail复用既有发送器/Pod API/独立D3与清理，不新增伸缩算法。人类证明/引用仍用户本人PENDING。正式72+6 NOT_RUN，工程门不构成科研端点通过。"]
    destination=ROOT/"docs/v3-sys-capacity.md"
    destination.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(destination.relative_to(ROOT),decision["status"],len(rows),"registered points")


def report_round_two():
    ledger=ROOT/"results/v3-sys-capacity-r2.csv"
    with ledger.open(encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    decision_path=WORK/"capacity-r2-decision.json"
    decision=json.loads(decision_path.read_text("utf-8")) if decision_path.exists() else dict(status="RUNNING",selected_capacity=None)
    lines=["# v3-sys 第二轮连接轮询容量校准", "", "日期：2026-10-25，Asia/Shanghai。", "",
        "用户授权T1：256独立Transport，每个1连接，测量前health预连接，计划request_id mod256分配；断开重拨单列。探针timeout2/failure3、period1/initial0。其余D系列、CPU工作/限额、SLO18.924ms和timeout189.2ms保持。", "",
        "4pod网格40/50/60/70，每点180秒：0坏窗、0失败、含零计数Pod的最大/均值≤1.3。最大合格及相邻档各追加3重复。最后8pod300秒，前180最多1/36坏窗，全程0失败，全程D10 p99≤1.8924ms且绑定2核CPU<30%。完整11次、第二轮总上限12，无第三轮；T2只在完整可观察的T1分发失败且预算足够时启用一次。",
        "", "| 编号 | 用途 | Pod×每Pod RPS | 状态 | 坏窗/非空窗 | 全程失败 | 分发最大/均值 | 发送p99 ms | CPU占绑定核% | SLO/D10 |",
        "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cpu=f"{float(r['sender_cpu_fraction'])*100:.3f}" if r['sender_cpu_fraction'] else "未产生"
        ratio=f"{float(r['backend_max_over_mean']):.4f}" if r['backend_max_over_mean'] else "未产生"
        late=f"{float(r['late_p99_ms']):.4f}" if r['late_p99_ms'] else "未产生"
        lines.append("|"+"|".join((r['label'],r['purpose'],r['pods']+"×"+r['per_pod_rps'],r['status'],
            r['slo_bad_windows']+"/"+r['slo_windows'],r['failed'],ratio,late,
            cpu,r['slo_pass']+"/"+r['fidelity_pass']))+"|")
    lines += ["", f"第二轮实际登记 **{len(rows)}次**；第一轮保留5次，两轮累计 **{5+len(rows)}次**。当前状态 **{decision['status']}**，选定c **{decision.get('selected_capacity')}**。",
        "", "该决策只指静态容量校准。它不自动解除独立源审指出的初始1Pod固定长连接风险，不代表新组件或正式78槽通过；相关实际运行状态另报。",
        "", "网格/重复中的D10列只描述；只有末次300秒才决定D10。不得用SLO通过替代未测的8c保真度，或把发送失败解释为服务容量的已隔离根因。未完成请求按失败处理，不剔除不利点。"]
    connection_summary=[]
    for r in rows:
        path=WORK/(r['label']+"-events.jsonl")
        events=[json.loads(s) for s in path.read_text('utf-8').splitlines()] if path.exists() else []
        connections=[e for e in events if e['type']=='connection']
        readiness=[e for e in events if e['type']=='transport_ready']
        success=sum(e['success'] for e in connections)
        redial=sum(e['redial'] for e in connections)
        failed=sum(not e['success'] for e in connections)
        connection_summary.append((r['label'],len(readiness),success,redial,failed))
        if (len(readiness),success,redial,failed)!=(1,256,0,0):
            lines += ['',f"{r['label']}连接记录：就绪{len(readiness)}、成功拨号{success}、重拨{redial}、拨号失败{failed}。"]
        if r['status']!='MEASURED': lines.append("该点无完整端点："+r['note'])
    lines += ['', f"连接记录汇总：{sum(r[1]==1 for r in connection_summary)}/{len(rows)}次有唯一预连接就绪记录；"
              f"成功拨号共{sum(r[2] for r in connection_summary)}，重拨{sum(r[3] for r in connection_summary)}，"
              f"拨号失败{sum(r[4] for r in connection_summary)}。原始事件、计划及逐请求连接/响应Pod留在本地work/v3-sys，不公开逐请求数组。"]
    if decision.get('next_stage')=='FINAL_FALLBACK_A':
        lines += ["", "第二轮未取得完整资格容量，按用户规则最终回A；不建立v3-sys-design-v2，不运行78个正式槽位，不再追加第三轮。系统比较写为局限/未来工作；原有一次合成实时组件的结论只涉及控制器与监视器实现。"]
    component_path=ROOT/'results/v3-sys-component-r2.csv'
    if component_path.exists():
        with component_path.open(encoding='utf-8') as stream: components=list(csv.DictReader(stream))
        lines += ['', '## 第二轮实时组件检查（非端点）', '',
                  '| 状态 | +2动作 | 在途禁缩tick | 缩容动作 | 五tick稳定 | 取消在途 | 监视标志 |',
                  '|---|---|---|---|---|---|---|']
        for row in components:
            lines.append('|'+ '|'.join(row.get(k,'') for k in ('status','plus_two_actions',
                'inflight_downscale_blocked_ticks','shrink_actions','five_tick_stability',
                'cancelled_inflight','monitor_flags'))+'|')
            path=WORK/(row['label']+'-events.jsonl'); requests_path=WORK/(row['label']+'-requests.csv')
            if path.exists() and requests_path.exists():
                records=[json.loads(s) for s in path.read_text('utf-8').splitlines()]
                with requests_path.open(encoding='utf-8') as stream: requests=list(csv.DictReader(stream))
                diagnostic=component_distribution(records,requests)
                lines += ['', f"同一次合成恒定负载的描述性分发核对：最多{diagnostic['maximum_ready']}个并发Ready；"
                    f"只取该Ready数量期间的{diagnostic['planned_requests']}个计划请求，各Pod实际响应数（含零）如下。"
                    '这不新增测量，不构成HPA或正式片段端点，也不改变组件原有判据。', '',
                    '| Pod | 响应数 |', '|---|---|']
                lines += [f'|{p}|{n}|' for p,n in sorted(diagnostic['pod_counts'].items())]
                lines += ['', f"缺失/不在当时Ready集合的响应：{diagnostic['missing_or_not_ready']}。"
                    '最大Ready期间的分发诊断与静态4/8Pod校准分开；实际建连时只有1Pod的事实不能由静态通过替代。']
    lines += ["", "独立复核见v3-sys-capacity-r2-review.md；同Codex协作体系，不等同外部审核或真实作者签核。", "",
        "技能实效：Ponytail复用既有HTTP/校准/组件路径；CCFA固定预算、选择与准入；scientific-critical-thinking区分分发/发送/服务因果与未完成证据；Nature统计保留每次运行而不把窗口或请求当独立样本作显著性推断。", "",
        "Transport参数与复用语义核对：[Go官方文档](https://pkg.go.dev/net/http#Transport)，本地固定Go1.27.3源transport.go的idleConn记录最近使用连接在列表末端。该机制解释为何单一大池不足以规定请求落点；256独立连接仍须实际分发验证。"]
    target=ROOT/"docs/v3-sys-capacity-r2.md"
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(target.relative_to(ROOT),decision['status'],len(rows))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--round2",action="store_true")
    report_round_two() if parser.parse_args().round2 else report()

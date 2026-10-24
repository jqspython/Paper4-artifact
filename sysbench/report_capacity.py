"""Render the bounded calibration and component evidence from retained CSVs."""
import csv
import json
from pathlib import Path

from sysbench.calibrate import ROOT, WORK, LEDGER


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


if __name__=="__main__": report()

"""Render the operational exploration review; use --ref to reproduce from Git CSVs."""

import argparse
import csv
import io
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASES = ("HPA", "Point+margin", "PAC-h", "BACC-cal")
VARIANTS = ("primary", "ITT", "tolerance_0.001", "tolerance_0.005", "tolerance_0.02",
            "mixed_median", "Huawei_source_slots", "total_cost", "validation_budget_met")


def render(rows, source):
    cells = {(r["variant"], r["scope"], r["metric"]): r for r in rows}
    if len(cells) != len(rows) or {r["stage"] for r in rows} != {"explore"}:
        raise ValueError("Unique exploration cells are required")
    def cell(v, k, s="overall"):
        return cells[v, s, k]
    def value(v, k, s="overall"):
        return cell(v, k, s)["value"]
    def percent(x):
        return f"{100 * float(x):.2f}%"
    def ratio(x):
        return f"{float(x):.6f}"
    selected = value("selection", "selected_eta")
    text = ["# v2-lite 探索结果：数值冻结前审核包", "",
            "日期：2026-10-21，Asia/Shanghai。**仅探索；尚未数值冻结或确证。**", "",
            f"数字来源：`{source}`。本文件由脚本读取完整 CSV 生成，不手工转录；它是运行审核报告，尚未把 v2 结果插入论文。旧 v1 失败及其标签不变。", "",
            "## 1. 运行与完整性", "",
            "原150应用（三来源各50）、四主delay、五η。150/150任务、15,000测试槽、9,000新增nominal候选、750锁点，共24,750行。HPA/Point/PAC旧450锁点资格逐项核验；新增BACC-cal和完整BACC-lite各30候选/应用。所有测试配置与各自nominal锁点相同；status均OK，UNAVAILABLE为0。", "",
            "12,000条wrapper的实际loss—λ末值恒等式全过（最大浮点差3.889e−13），其中10,800条契约内上界/最小λ检查全过。它们不是独立应用数，不据零越界宣称界紧。", "",
            "| 来源 | 选中应用 | 每个主用例契约内 | 契约外比例 |", "|---|---:|---:|---:|"]
    for s in ("huawei_private2023", "huawei_public2023", "azure2019"):
        n = int(float(value("support", "contract_apps", s + "/delay120")))
        if any(int(float(value("support", "contract_apps", s + "/" + case))) != n
               for case in ("delay180", "delay300", "delay600")):
            raise ValueError("This report's equal-delay support summary no longer applies")
        text.append(f"| {s} | 50 | {n} | {percent(1 - n / 50)} |")
    text += ["", "每来源×每主例均≥30。权重为Private/Public/Azure=1/4、1/4、1/2；下文不将135个契约应用简单混合成总体样本。", "",
             "## 2. 五η的目标与安全点", "",
             "成本目标是四个wrapper相对各自基础的来源中位成本比的厂商/类别宏平均。安全筛选先要求宏合规≥90%、每wrapper≥85%，再最小化该目标，1e−10内取较小η。合规按实际容量违约率≤6%定义，100%合规不等于零违约。", "",
             "| η | 目标成本比 | 宏合规 | HPA封装 | Point封装 | PAC封装 | BACC-cal封装 | 可选 | 选中 |",
             "|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for eta in ("0.02", "0.05", "0.1", "0.2", "0.5"):
        v = "eta_" + eta
        safety = " | ".join(percent(value(v, "wrapper_" + b)) for b in BASES)
        text.append(f"| {eta} | {ratio(value(v, 'all_cost_ratio'))} | {percent(value(v, 'p3_compliance'))} | {safety} | {'是' if value(v, 'selection_eligible') == '1' else '否'} | {'是' if value(v, 'selected') == '1' else '否'} |")
    text += ["", f"**冻结规则推选η={selected}**；不是边界点，不扩网格。该值只在结果CSV登记，配置`selected_eta`仍为−1，确认两个审核字段仍PENDING，供用户数值审核后才冻结。", "",
             "## 3. 所选点的探索诊断", "",
             "以下为选η所用探索集上的诊断，不是新应用确证的P1–P3裁定。区间没有校正同集选η，不能拿来宣称确证成功。", "",
             "| 指标 | 点估计 | 来源内应用配对bootstrap 95%区间 |",
             "|---|---:|---:|"]
    for key, label, fmt in (("p1_delta", "P1 U类合规差（百分点）", lambda x: f"{100 * float(x):.3f}"),
                            ("p1_cost_ratio", "U类主成本比（护栏1.25）", ratio),
                            ("p2_cost_ratio", "P2 O类主成本比", ratio),
                            ("p2_delta", "O类合规差（百分点）", lambda x: f"{100 * float(x):.3f}"),
                            ("p3_compliance", "P3 四wrapper宏合规", percent)):
        r = cell("primary", key)
        text.append(f"| {label} | {fmt(r['value'])} | [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}] |")
    text += ["", "主分析固定厂商及来源、每来源重抽应用2000次、seed20261020；四wrapper各自宏合规均100%。探索脚本的PASS标签仅表示这些探索数值达到设计线，不能替代用户审核或新应用确证。", "",
             "| 厂商 | U类合规差（百分点） | U主成本比 | O主成本比 | O合规差（百分点） | 四wrapper宏合规 |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in ("Huawei", "Azure"):
        text.append(f"| {s} | {100 * float(value('primary', 'p1_delta', s)):.3f} | {ratio(value('primary', 'p1_cost_ratio', s))} | {ratio(value('primary', 'p2_cost_ratio', s))} | {100 * float(value('primary', 'p2_delta', s)):.3f} | {percent(value('primary', 'p3_compliance', s))} |")
    text += ["", "两厂商U类合规提升均>0、O类主成本比均<1。P2仅表明既定合规约束下预算回收，不能称同实际风险成本优势。O类为PAC-h与关闭内部pacing的BACC-cal；完整BACC-lite只作另列竞争者，不能混称。", "",
             "## 4. 九变体一次性完整报告", "",
             "| 变体 | U合规差（百分点） | U成本比 | O成本比 | O合规差（百分点） | 宏合规 |",
             "|---|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        text.append(f"| {v} | {100 * float(value(v, 'p1_delta')):.3f} | {ratio(value(v, 'p1_cost_ratio'))} | {ratio(value(v, 'p2_cost_ratio'))} | {100 * float(value(v, 'p2_delta')):.3f} | {percent(value(v, 'p3_compliance'))} |")
    text += ["", "total_cost列使用实际付费分钟的分子/分母求比，其总体再等权平均两个厂商比值；其余按各变体预定的中位函数。U类主中位成本比<1，但实际总付费比约1.029，不能据此说全部支出下降。ITT撤需求契约mask，不撤基础可用性规则；敏感性不改变主选择。", "",
             "| 厂商 | U总付费比 | U应用比P90 | O总付费比 | O应用比P90 |",
             "|---|---:|---:|---:|---:|"]
    for s in ("Huawei", "Azure"):
        vals = [ratio(value("total_cost", k, s)) for k in ("p1_cost_ratio", "p1_app_total_ratio_p90", "p2_cost_ratio", "p2_app_total_ratio_p90")]
        text.append(f"| {s} | " + " | ".join(vals) + " |")
    text += ["", "F1：所有source×base UNAVAILABLE=0，没有触发>5%门。四主base的验证预算未达标配对共10项（旧9+新BACC-cal1）；只在validation_budget_met变体排该配对，其他base保留。其中Private四个配对已在主契约外，故此变体相对主分析实际额外排除6个契约内HPA配对（Public4、Azure2），不能把10个标记都称额外删除。完整BACC-lite另有1项，不计入四主base的排除数。每source×base可用数、未达标数和全部次表均在同一endpoints CSV。", "",
             "## 5. 工程、审阅与下一边界", "",
             "正式控制运行约21分42秒（SQLite创建08:56:05至最终summary导出09:17:47）；正式九变体分析约63秒。硬件/人工微基准的保守预算仍见冻结工作量CSV：两阶段控制合计74.52小时、另分析不足0.11小时；不把当前实际耗时当确认时间承诺。", "",
             "冻结标签`v2-design-v1`指向4373a4ed，关键源码/配置/协议未改。正式分析仅用canonical命令`python -B -X utf8 scripts/analyze_v2_lite.py explore`，入口外核配置对象==冻结TOML，未传替代--config。没有抽取确认ID或访问确认正文，Alibaba、kind、ACI新增及Q均未启动。", "",
             "[冻结协议](v2-protocol.md) · [工作量与性质核查](v2-design-review.md) · [选参流水线](v2-lite-pipeline.md) · [跨作者及独立结果复算](v2-lite-exploration-review.md) · [完整端点CSV](../results/v2-lite-endpoints.csv) · [完整运行聚合CSV](../results/v2-lite-summary.csv)", "",
             "[六章英文源稿](../manuscript/main.md)与[去注释阅读副本](../manuscript/submission.md)保留v1负结果，没有插入v2摘要/结论。16条来源核读、Claude版本未记录及选题/查新/协议和稿件审阅用途、11月1日前人类作者核验安排见[文献与作者核验表](v2-literature-verification.md)；AI检查不计成人类签核。", "",
             "本轮回传供用户数值冻结前审核。收到数值批准并建立`v2-confirm-v1`后才能准备新应用确证；当前不自行推进。"]
    return "\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="immutable Git reference containing the reviewed endpoint CSV")
    args = parser.parse_args()
    name = "results/v2-lite-endpoints.csv"
    text = (subprocess.check_output(["git", "show", f"{args.ref}:{name}"], cwd=ROOT, encoding="utf-8")
            if args.ref else (ROOT / name).read_text("utf-8-sig"))
    rows = list(csv.DictReader(io.StringIO(text)))
    source = f"{args.ref}:{name}" if args.ref else name + " (completed, reviewed exploration export)"
    destination = ROOT / "docs/v2-lite-exploration.md"
    destination.write_text(render(rows, source), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

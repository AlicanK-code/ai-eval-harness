"""Self-contained HTML reports (no external assets) and plain-text console summaries."""

from __future__ import annotations

import json
import math
from html import escape
from typing import Any

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;--accent:#3b5bdb;
--good:#2b8a3e;--bad:#c92a2a;--warn:#b7791f;--bar:#dbe4ff;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1e1e1c;--ink:#ecebe6;--muted:#9a9993;
--line:#33332f;--accent:#7c95ff;--good:#69db7c;--bad:#ff8787;--warn:#ffd43b;--bar:#2b3566}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 12px}
.sub{color:var(--muted);font-size:13px}.mono{font-family:var(--mono);font-size:12.5px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:24px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums}
.card .n{color:var(--muted);font-size:12.5px}
table{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top;font-variant-numeric:tabular-nums}
th{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
tr:last-child td{border-bottom:0}
.bar{position:relative;height:18px;background:var(--line);border-radius:4px;min-width:120px}
.bar span{position:absolute;inset:0 auto 0 0;background:var(--accent);border-radius:4px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.pill{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line);margin-right:4px}
details summary{cursor:pointer;list-style:none}details summary::-webkit-details-marker{display:none}
details[open] summary{font-weight:600}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);
border-radius:8px;padding:10px;margin:6px 0;max-height:320px;overflow:auto;font:12.5px/1.45 var(--mono)}
.controls{display:flex;gap:12px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
input[type=search]{flex:1;min-width:180px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;
background:var(--panel);color:var(--ink)}
.table-wrap{overflow-x:auto}
"""

FILTER_JS = """
const q=document.getElementById('q'),f=document.getElementById('failing');
function apply(){const t=q.value.toLowerCase();document.querySelectorAll('tr[data-case]').forEach(r=>{
const m=r.dataset.case.toLowerCase().includes(t)||r.dataset.tags.toLowerCase().includes(t);
const bad=parseFloat(r.dataset.mean)<1||r.dataset.mean==='';r.style.display=(m&&(!f.checked||bad))?'':'none'})}
q.addEventListener('input',apply);f.addEventListener('change',apply);
"""


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.{digits}f}%"


def _usd(x: float | None) -> str:
    if x is None:
        return "unknown"
    return f"${x:.4f}" if x < 1 else f"${x:.2f}"


def _secs(x: float) -> str:
    return "-" if math.isnan(x) else f"{x:.2f}s"


def _card(k: str, v: str, n: str = "") -> str:
    return f'<div class="card"><div class="k">{escape(k)}</div><div class="v">{v}</div><div class="n">{n}</div></div>'


def _bar(x: float) -> str:
    w = 0 if math.isnan(x) else max(0.0, min(1.0, x)) * 100
    return f'<div class="bar"><span style="width:{w:.1f}%"></span></div>'


def _page(title: str, body: str, script: str = "") -> str:
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main><script>{script}</script></body></html>"
    )


def _status_pills(status: dict[str, int]) -> str:
    cls = {"ok": "good", "truncated": "warn", "refused": "bad", "incomplete": "warn"}
    return "".join(f'<span class="pill {cls.get(k, "")}">{escape(k)}: {v}</span>' for k, v in sorted(status.items()))


def render_run(s: dict[str, Any]) -> str:
    h = s["headline"]
    c = s["counts"]
    errors = c["unresolved_errors"]
    cards = "".join([
        _card(f"{s['primary_grader']} (mean)", _pct(h["mean"]), f"95% CI {_pct(h['ci_low'])} – {_pct(h['ci_high'])}"),
        _card("Pass rate", _pct(h["pass_rate"]), f"strict mean {_pct(h['strict_mean'])}"),
        _card("Noise floor", f"±{_pct(s['noise_floor'])}", "smaller differences aren't resolvable"),
        _card("Model cost", _usd(s["cost"]["model_usd"]), f"judge {_usd(s['cost']['judge_usd'])}"),
        _card("Latency p50", _secs(s["latency_s"]["p50"]), f"p95 {_secs(s['latency_s']['p95'])}"),
        _card("Rows", f"{c['rows']}/{c['expected_rows']}",
              f"{sum(errors.values())} unresolved errors · {c['retried_rows']} retried"),
    ])

    grader_rows = "".join(
        f"<tr><td>{escape(name)}{' <span class=pill>primary</span>' if name == s['primary_grader'] else ''}</td>"
        f"<td>{_pct(m['mean'])}</td><td>{_pct(m['ci_low'])} – {_pct(m['ci_high'])}</td>"
        f"<td>{_pct(m['pass_rate'])}</td><td>{_pct(m['answered_rate'])}</td></tr>"
        for name, m in s["metrics"].items()
    )
    tag_rows = "".join(
        f"<tr><td>{escape(tag)}</td><td>{t['n_cases']}</td><td style='width:50%'>{_bar(t['mean'])}</td>"
        f"<td>{_pct(t['mean'])}</td></tr>"
        for tag, t in s["by_tag"].items()
    )

    case_rows = []
    for case in sorted(s["cases"], key=lambda c: (c["mean"] if c["mean"] is not None else -1, c["case_id"])):
        reps_html = []
        for r in case["reps"]:
            score = "–" if r["score"] is None else f"{r['score']:.2f}"
            mark = "good" if r["passed"] else "bad"
            reps_html.append(
                f"<div><span class='pill {mark}'>rep {r['rep']} · {escape(r['status'])} · score {score}</span>"
                f"<span class='sub'>{r['attempts']} attempt(s) · {r['latency_s']:.2f}s</span>"
                f"<pre>{escape(r['output']) or '<em>(empty)</em>'}</pre>"
                f"<pre>{escape(json.dumps(r['grades'], indent=2, ensure_ascii=False))}</pre></div>"
            )
        mean_txt = "" if case["mean"] is None else f"{case['mean']:.3f}"
        cls = "" if case["mean"] is None else ("good" if case["mean"] == 1 else "bad" if case["mean"] < 0.5 else "warn")
        case_rows.append(
            f"<tr data-case='{escape(case['case_id'])}' data-tags='{escape(' '.join(case['tags']))}' data-mean='{mean_txt}'>"
            f"<td><details><summary><span class='mono'>{escape(case['case_id'])}</span> "
            f"{''.join(f'<span class=pill>{escape(t)}</span>' for t in case['tags'])}</summary>"
            f"<div class='sub'>Prompt</div><pre>{escape(case['prompt'])}</pre>{''.join(reps_html)}</details></td>"
            f"<td class='{cls}'>{mean_txt or '–'}</td></tr>"
        )

    error_html = ""
    if errors:
        error_html = (
            "<h2>Unresolved errors</h2><p class='sub'>These attempts produced no scorable output and are "
            "excluded from every score. Re-run with <span class='mono'>--resume</span> to retry them.</p>"
            + "".join(f"<span class='pill bad'>{escape(k)}: {v}</span>" for k, v in errors.items())
        )

    body = f"""
<h1>{escape(s['suite'])}</h1>
<div class="sub">{escape(s['description'])}</div>
<div class="sub mono" style="margin-top:6px">{escape(s['provider'])}:{escape(s['model'])} · effort {escape(str(s['effort']))}
 · {s['reps']} reps · {escape(s['created_at'])} · dataset {escape(s['dataset_sha256'][:12])}</div>
<div class="cards">{cards}</div>
<div style="margin-top:12px">{_status_pills(c['status'])}
<span class="sub">refusal rate {_pct(c['refusal_rate'])} · truncation rate {_pct(c['truncation_rate'])}
 — scores use status=ok rows only; "strict" counts every other status as 0.</span></div>
{error_html}
<h2>Graders</h2><div class="table-wrap"><table><tr><th>Grader</th><th>Mean</th><th>95% CI</th><th>Pass rate</th><th>Answered</th></tr>{grader_rows}</table></div>
<h2>By tag ({escape(s['primary_grader'])})</h2><div class="table-wrap"><table><tr><th>Tag</th><th>Cases</th><th></th><th>Mean</th></tr>{tag_rows}</table></div>
<h2>Cases</h2>
<div class="controls"><input id="q" type="search" placeholder="Filter by case id or tag">
<label class="sub"><input id="failing" type="checkbox"> only imperfect cases</label></div>
<div class="table-wrap"><table><tr><th>Case (worst first)</th><th>Mean</th></tr>{''.join(case_rows)}</table></div>
"""
    return _page(f"{s['suite']} · {s['model']}", body, FILTER_JS)


def render_compare(cmp: dict[str, Any]) -> str:
    a, b, d = cmp["a"], cmp["b"], cmp["diff"]
    delta = d["mean_diff"]
    verdict_cls = "good" if cmp["significant"] and delta > 0 else "bad" if cmp["significant"] else "warn"
    verdict = (
        f"B is better by {delta * 100:.1f} pts" if cmp["significant"] and delta > 0
        else f"B is worse by {-delta * 100:.1f} pts" if cmp["significant"]
        else "No significant difference"
    )
    label = lambda s: f"{s['provider']}:{s['model']}"  # noqa: E731
    cards = "".join([
        _card("Verdict", f"<span class='{verdict_cls}'>{escape(verdict)}</span>",
              f"B − A, paired 95% CI {d['ci_low'] * 100:+.1f} to {d['ci_high'] * 100:+.1f} pts over {d['n']} cases"),
        _card("A · " + cmp["primary_grader"], _pct(a["headline"]["mean"]), escape(label(a))),
        _card("B · " + cmp["primary_grader"], _pct(b["headline"]["mean"]), escape(label(b))),
        _card("Cost A → B", f"{_usd(a['cost']['model_usd'])} → {_usd(b['cost']['model_usd'])}", "model spend, whole run"),
        _card("Latency p50 A → B", f"{_secs(a['latency_s']['p50'])} → {_secs(b['latency_s']['p50'])}",
              f"p95 {_secs(a['latency_s']['p95'])} → {_secs(b['latency_s']['p95'])}"),
    ])
    tags = sorted(set(a["by_tag"]) | set(b["by_tag"]))
    nan = {"mean": float("nan")}
    tag_rows = "".join(
        f"<tr><td>{escape(t)}</td><td>{_pct(a['by_tag'].get(t, nan)['mean'])}</td>"
        f"<td>{_pct(b['by_tag'].get(t, nan)['mean'])}</td></tr>"
        for t in tags
    )
    a_cases = {c["case_id"]: c["mean"] for c in a["cases"]}
    b_cases = {c["case_id"]: c["mean"] for c in b["cases"]}
    shared = [k for k in a_cases if k in b_cases and a_cases[k] is not None and b_cases[k] is not None]
    shared.sort(key=lambda k: b_cases[k] - a_cases[k])
    case_rows = "".join(
        f"<tr><td class='mono'>{escape(k)}</td><td>{a_cases[k]:.2f}</td><td>{b_cases[k]:.2f}</td>"
        f"<td class='{'good' if b_cases[k] > a_cases[k] else 'bad' if b_cases[k] < a_cases[k] else ''}'>"
        f"{b_cases[k] - a_cases[k]:+.2f}</td></tr>"
        for k in shared
    )
    body = f"""
<h1>{escape(a['suite'])}: A vs B</h1>
<div class="sub mono">A = {escape(a['run_dir'])}<br>B = {escape(b['run_dir'])}</div>
<div class="cards">{cards}</div>
<p class="sub">{len(cmp['flips']['improved'])} case(s) improved by ≥0.5, {len(cmp['flips']['regressed'])} regressed by ≥0.5.
Pairing by case removes per-case difficulty from the comparison.</p>
<h2>By tag</h2><div class="table-wrap"><table><tr><th>Tag</th><th>A</th><th>B</th></tr>{tag_rows}</table></div>
<h2>Per case (biggest regressions first)</h2>
<div class="table-wrap"><table><tr><th>Case</th><th>A</th><th>B</th><th>Δ</th></tr>{case_rows}</table></div>
"""
    return _page(f"Compare · {a['suite']}", body)


def console_summary(s: dict[str, Any]) -> str:
    h, c = s["headline"], s["counts"]
    lines = [
        f"{s['suite']}  |  {s['provider']}:{s['model']}  |  {s['reps']} reps",
        f"  {s['primary_grader']:<18} {_pct(h['mean'])}  (95% CI {_pct(h['ci_low'])} to {_pct(h['ci_high'])}, "
        f"noise floor +/-{_pct(s['noise_floor'])})",
        f"  pass rate          {_pct(h['pass_rate'])}   strict mean {_pct(h['strict_mean'])}",
    ]
    for name, m in s["metrics"].items():
        if name != s["primary_grader"]:
            lines.append(f"  {name:<18} {_pct(m['mean'])}  (pass {_pct(m['pass_rate'])})")
    lines += [
        f"  status             {', '.join(f'{k}={v}' for k, v in sorted(c['status'].items()))}",
        f"  unresolved errors  {sum(c['unresolved_errors'].values())} {c['unresolved_errors'] or ''}".rstrip(),
        f"  cost               model {_usd(s['cost']['model_usd'])}, judge {_usd(s['cost']['judge_usd'])}",
        f"  latency            p50 {_secs(s['latency_s']['p50'])}, p95 {_secs(s['latency_s']['p95'])}",
    ]
    return "\n".join(lines)


def console_compare(cmp: dict[str, Any]) -> str:
    d = cmp["diff"]
    a, b = cmp["a"], cmp["b"]
    verdict = "SIGNIFICANT" if cmp["significant"] else "not significant"
    return "\n".join([
        f"A {a['provider']}:{a['model']:<20} {_pct(a['headline']['mean'])}",
        f"B {b['provider']}:{b['model']:<20} {_pct(b['headline']['mean'])}",
        f"B - A = {d['mean_diff'] * 100:+.1f} pts  (paired 95% CI {d['ci_low'] * 100:+.1f} to {d['ci_high'] * 100:+.1f}, "
        f"n={d['n']} cases) -> {verdict}",
        f"improved: {', '.join(cmp['flips']['improved']) or '-'}",
        f"regressed: {', '.join(cmp['flips']['regressed']) or '-'}",
    ])

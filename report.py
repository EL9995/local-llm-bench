#!/usr/bin/env python3
"""Turn a saved run into something readable: a Markdown file or a standalone HTML page.

Each mode is laid out as a list of blocks (headings, text, tables, photos), and the
same blocks render to either format.

Run:   python3 report.py runs/<file>.json     writes runs/reports/<file>.md and .html
"""

import base64
import html
import json
import sys
from pathlib import Path

import relay

FMT = {"text": "plain text", "markdown": "Markdown", "json": "JSON", "fields": "custom fields"}
STRAT = {"together": "all at once", "batch": "batch", "stream": "photo by photo"}


def show(v):
    return ", ".join(map(str, v)) if isinstance(v, list) else str(v if v is not None else "")


# ---- blocks: ("h2", text) ("p", text) ("text", label, body) ("table", headers, rows) ("img", name) ("note", text)

def blocks(r):
    mode = r["mode"]
    out = [("h1", r["task"] or {"vision": "Reading a photo"}.get(mode, mode.title())),
           ("meta", [("Mode", mode.title()), ("When", r.get("time", "").replace("T", " ")),
                     ("Models", " → ".join(r["chain"])), ("Took", f"{r['secs']} s, {r['calls']} model calls")]
            + ([("Rule for every model", r["rule"])] if r.get("rule") else []))]
    out += {"relay": relay_blocks, "blind": blind_blocks, "orchestrator": orchestrator_blocks,
            "vision": vision_blocks, "roundtable": roundtable_blocks}.get(mode, steps_blocks)(r)
    out.append(("note", f"Saved run {r.get('id', '')} · Local LLM Bench"))
    return out


def meta(t):
    return f"{t.get('secs', 0)} s · {t.get('tok_s', 0)} tok/s" + (" · cut off at the length limit" if t.get("cut") else "")


def relay_blocks(r):
    out = [("h2", "Final answer"), ("p", r["answer"])]
    s = r.get("sabotage")
    if s:
        verdict = {"caught": f"Caught by layer {s.get('caught_by', 0) + 1} ({s.get('caught_by_model')}).",
                   "slipped": f"Slipped through: nobody above layer {s['layer'] + 1} changed it.",
                   "no_lie": "The saboteur didn't lie; it gave the honest answer anyway."}[s["outcome"]]
        out += [("h2", "Saboteur"), ("p", f"Layer {s['layer'] + 1} ({s['model']}) was told to plant a wrong answer. "
                                          f"Honest answer: {s.get('honest')}. Planted: {s.get('planted')}. {verdict}")]
    out.append(("h2", "Step by step"))
    what = {"down": "told the model below", "answer": "answered", "up": "checked the answer and passed up"}
    for t in r["trace"]:
        label = what[t["dir"]]
        if t["dir"] == "up" and t.get("changed"):
            label = "changed the answer"
        if t.get("sabotage"):
            label = "planted a wrong answer" if t.get("lied", True) else "was told to lie, answered honestly"
        out.append(("text", f"Layer {t['layer'] + 1} · {t['model']} {label} ({meta(t)})", t["sent"]))
    return out


def blind_blocks(r):
    out = [("h2", "Each model's answer, on its own")]
    for t in r["trace"]:
        out.append(("text", f"{t['model']} ({meta(t)})", t["sent"]))
    return out


def orchestrator_blocks(r):
    plan, *rest = r["trace"]
    merge = next((t for t in rest if t["dir"] == "merge"), {})
    cut = " (cut off at the token limit)" if merge.get("cut") else ""
    out = [("h2", "Final answer"), ("text", f"Merged by {plan['model']}{cut}", r["answer"])]
    out += [("h2", f"The plan: {len(r['subtasks'])} subtasks"), ("text", f"{plan['model']} ({meta(plan)})", plan["sent"]),
            ("h2", "The workers")]
    for t in (x for x in rest if x["dir"] == "work"):
        out.append(("text", f"Worker {t['layer']} · {t['model']} ({meta(t)}) · subtask: {t['task']}", t["sent"]))
    return out


def vision_blocks(r):
    names = r.get("images") or ([r["image"]] if r.get("image") else [])
    out = [("h2", "Photos")] + [("img", n) for n in names]
    out.append(("p", f"Read by {', '.join(r['readers'])} as {FMT.get(r['format'], r['format'])}"
                     + (f", {STRAT[r['strategy']]}" if len(names) > 1 else "")
                     + (f". Final agent: {r['combiner']}." if r["format"] != "fields" or r["answer"] != r.get("combined") else ".")
                     + (f" First result at {r['first_result']} s, {r['load_secs']} s loading models." if r.get("first_result") else "")))
    if r["answer"] != r.get("combined"):
        out += [("h2", "Final agent's answer"), ("p", r["answer"])]
    if r["format"] == "fields":
        for p, grid in enumerate(r.get("grids", [])):
            out.append(("h2", "Field by field" + (f": photo {p + 1}" if len(r["grids"]) > 1 else "")))
            rows = []
            for name, g in grid.items():
                cells = [show(g["values"][m]) or "—" for m in r["readers"]]
                cells = [c if g["match"][m] else f"{c} ✗" for c, m in zip(cells, r["readers"])]
                vote = f"{show(g['chosen'])} ({g['agree']}/{len(r['readers'])}{', tie' if g['tie'] else ''})" if g["found"] \
                    else "not in the photo"
                rows.append([name, *cells, vote])
            out.append(("table", ["Field", *r["readers"], "Most common"], rows))
    else:
        out.append(("text", "Combined text", r.get("combined", "")))
    out += [("h2", "How each model did"),
            ("table", ["Model", "Matches the combined result"],
             [[a["model"], f"{round(a['match'] * 100)}%"] for a in r.get("agreement", [])])]
    if r["format"] != "fields":
        out.append(("h2", "Each reading"))
        for t in r["trace"]:
            if t["dir"] == "extract":
                photo = "all photos" if t.get("photo") is None and len(names) > 1 else \
                    f"photo {t['photo'] + 1}" if t.get("photo") is not None and len(names) > 1 else "the photo"
                out.append(("text", f"{t['model']} read {photo} ({meta(t)})", t["sent"]))
    return out


def roundtable_blocks(r):
    out = []
    if r.get("winners"):
        w = r["winners"]
        out += [("h2", "Winner" if len(w) == 1 else "Tie"), ("p", "  /  ".join(f"{m}: {r['answers'][m]}" for m in w)),
                ("table", ["Model", "Votes"], [[m, str(c)] for m, c in r["tally"].items()])]
    if r.get("summary"):
        out += [("h2", f"Moderator ({r['moderator']})"), ("p", r["summary"])]
    out.append(("p", f"{r['rounds']} discussion round(s), {r.get('minds_changed', 0)} change(s) of mind."))
    last = None
    for t in r["trace"]:
        if t["dir"] == "say":
            if t["round"] != last:
                last = t["round"]
                out.append(("h2", "Round 0: everyone answers alone" if last == 0 else f"Round {last}: discussion"))
            changed = " · changed their mind" if t.get("changed") else ""
            out.append(("text", f"{t['model']}{changed} ({meta(t)})", t["sent"]))
    ballots = r.get("ballots") or []
    if ballots:
        out += [("h2", "The vote"), ("table", ["Voter", "Voted for", "Reason"],
                                     [[b["voter"], b["for"] or "spoiled ballot", b["reason"]] for b in ballots])]
    return out


def steps_blocks(r):
    return [("h2", "Answer"), ("p", show(r.get("answer")))] + \
           [("text", f"{t.get('model')} · {t.get('dir')}", t.get("sent", "")) for t in r["trace"]]


# ---- rendering ----

def to_markdown(r):
    out = []
    for b in blocks(r):
        kind = b[0]
        if kind == "h1":
            out.append(f"# {b[1]}\n")
        elif kind == "meta":
            out.append("\n".join(f"- **{k}:** {v}" for k, v in b[1]) + "\n")
        elif kind == "h2":
            out.append(f"## {b[1]}\n")
        elif kind == "p":
            out.append(f"{b[1]}\n")
        elif kind == "text":
            body = "\n".join("> " + line for line in str(b[2]).splitlines()) or ">"
            out.append(f"**{b[1]}**\n\n{body}\n")
        elif kind == "table":
            cell = lambda c: str(c).replace("|", "\\|").replace("\n", " ")
            out.append("| " + " | ".join(map(cell, b[1])) + " |\n|" + "---|" * len(b[1]) + "\n"
                       + "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in b[2]) + "\n")
        elif kind == "img":
            out.append(f"![photo](../images/{b[1]})\n")  # relative to runs/reports/
        elif kind == "note":
            out.append(f"---\n_{b[1]}_\n")
    return "\n".join(out)


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>
:root {{ --bg:#fbfaf7; --ink:#1d1d1b; --muted:#6b6a64; --line:#e2e0d8; --soft:#f1efe8; --accent:#2f6fdb; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --ink:#ecebe6; --muted:#9a998f; --line:#34332f;
  --soft:#222220; --accent:#6c9cf0; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
main {{ max-width:760px; margin:0 auto; padding:40px 16px 80px; }}
h1 {{ font-size:26px; line-height:1.25; margin:0 0 12px; }}
h2 {{ font-size:18px; margin:32px 0 10px; padding-top:16px; border-top:1px solid var(--line); }}
dl {{ display:grid; grid-template-columns:auto 1fr; gap:2px 14px; margin:0; font-size:14px; }}
dt {{ color:var(--muted); }} dd {{ margin:0; }}
.block {{ margin:14px 0; }} .block b {{ display:block; font-size:13px; color:var(--muted); margin-bottom:4px; font-weight:600; }}
.block div {{ background:var(--soft); border-left:3px solid var(--accent); padding:10px 14px; border-radius:6px;
  white-space:pre-wrap; word-break:break-word; }}
p {{ white-space:pre-wrap; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }}
th, td {{ text-align:left; vertical-align:top; padding:6px 8px; border-bottom:1px solid var(--line); }}
th {{ font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted); }}
img {{ max-width:100%; max-height:420px; border-radius:8px; border:1px solid var(--line); display:block; margin:8px 0; }}
.note {{ color:var(--muted); font-size:13px; margin-top:40px; }}
@media print {{ body {{ background:#fff; color:#000; }} h2 {{ break-after:avoid; }} .block {{ break-inside:avoid; }} }}
</style></head><body><main>
{body}
</main></body></html>"""


def to_html(r):
    e = lambda t: html.escape(str(t))
    out = []
    for b in blocks(r):
        kind = b[0]
        if kind == "h1":
            out.append(f"<h1>{e(b[1])}</h1>")
        elif kind == "meta":
            out.append("<dl>" + "".join(f"<dt>{e(k)}</dt><dd>{e(v)}</dd>" for k, v in b[1]) + "</dl>")
        elif kind == "h2":
            out.append(f"<h2>{e(b[1])}</h2>")
        elif kind == "p":
            out.append(f"<p>{e(b[1])}</p>")
        elif kind == "text":
            out.append(f'<div class="block"><b>{e(b[1])}</b><div>{e(b[2])}</div></div>')
        elif kind == "table":
            out.append("<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in b[1]) + "</tr>"
                       + "".join("<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>" for row in b[2]) + "</table>")
        elif kind == "img":
            path = relay.RUNS / "images" / b[1]
            if path.is_file():  # embedded, so the page works on its own
                out.append(f'<img alt="photo" src="data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}">')
        elif kind == "note":
            out.append(f'<p class="note">{e(b[1])}</p>')
    return PAGE.format(title=e(blocks(r)[0][1])[:80], body="\n".join(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.strip().splitlines()[-1])
    run = json.loads(Path(sys.argv[1]).read_text())
    folder = relay.RUNS / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / Path(sys.argv[1]).stem
    stem.with_suffix(".md").write_text(to_markdown(run))
    stem.with_suffix(".html").write_text(to_html(run))
    print(f"wrote {stem}.md and {stem}.html")

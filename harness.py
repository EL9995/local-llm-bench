#!/usr/bin/env python3
"""Score models on a suite of questions with known answers.

Every case in the suite runs against every model in the chain, each on its own.
Code checks every answer, so no model grades another. Results go to
runs/harness/ and a scorecard is printed at the end.

The orchestrator suite (a suite with "kind": "orchestrator") scores something
else: the first model splits each task into subtasks, the rest of the chain does
them, and the first model merges the results. Code scores the split, every
worker's answer, the orchestrator's merge, and a merge done by code from the
same worker answers, next to each model answering the whole task alone.

Run:   python3 harness.py                          (suites/basics.json, default chain)
       python3 harness.py orchestrator             (orchestrator, then workers: granite4.2:8b,gemma3:1b)
       python3 harness.py --chain granite4.2:8b,gemma3:1b --only mul,strawberry
       python3 harness.py --show runs/harness/<file>.json   (reprint a scorecard)
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import bench

SUITES = Path(__file__).with_name("suites")
OUT = bench.RUNS / "harness"
NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUMBER_WORDS.update(once=1, twice=2, thrice=3)


# ---- checks: each returns (passed, why) ----

def numbers(text):
    """Every number in the text, digits or words up to twenty, in order."""
    found = []
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?|[a-z]+", text.lower()):
        tok = m.group()
        if tok[0].isdigit():
            found.append(float(tok.replace(",", "")))
        elif tok in NUMBER_WORDS:
            found.append(float(NUMBER_WORDS[tok]))
    return found


def check_number(answer, want, prompt):
    """Right if the answer ends on the expected number ("144 / 12 = 12"), the first number
    it adds beyond the prompt's is the expected one ("5 cents ($0.05)"), or the number
    right after the first "is", "=" or "equals" is ("is 12, with a remainder of 0")."""
    nums = numbers(answer)
    if not nums:
        return False, "no number in the answer"
    given = set(numbers(prompt))
    new = [n for n in nums if n not in given]
    stated = re.search(r"(?:\bis\b|=|\bequals\b)\W{0,3}([\d,]+(?:\.\d+)?|[a-z]+)", answer.lower())
    said = numbers(stated.group(1)) if stated else []
    ok = nums[-1] == want or (bool(new) and new[0] == want) or (bool(said) and said[0] == want)
    return ok, f"got {', '.join(f'{n:g}' for n in (new or nums)[:4])}"


def word_count(text):
    return len(re.findall(r"[\w'-]+", text))


def as_list(v):
    return v if isinstance(v, list) else [v]


CHECKS = {
    "number": lambda a, v, c: check_number(a, float(v), c["prompt"]),
    "contains": lambda a, v, c: (any(x.lower() in a.lower() for x in as_list(v)),
                                 f"needs {' or '.join(as_list(v))}"),
    "not_contains": lambda a, v, c: (not any(x.lower() in a.lower() for x in as_list(v)),
                                     f"must not contain {' or '.join(as_list(v))}"),
    "regex": lambda a, v, c: (bool(re.search(v, a)), f"must match /{v}/"),
    "max_words": lambda a, v, c: (word_count(a) <= v, f"{word_count(a)} words, limit {v}"),
    "intact": lambda a, v, c: (c.get("_intact") is v, "message arrived intact" if v else ""),
}


def score(case, answer, record):
    answer = answer.replace("**", "").strip()  # markdown bold isn't part of the answer
    ctx = {**case, "_intact": record.get("intact")}
    fails = []
    for kind, want in case["check"].items():
        ok, why = CHECKS[kind](answer, want, ctx)
        if not ok:
            fails.append(f"{kind}: {why}")
    return not fails, fails


# ---- orchestrator suite: score the split, the workers and the merge ----

ORCH_CHAIN = "granite4.2:8b,gemma3:1b"
FLAGS = re.I | re.M


def find(key, text, last=False):
    """Where key first (or last) matches in text, or None. An empty key matches at the start."""
    if not key:
        return 0
    hits = [m.start() for m in re.finditer(key, text, FLAGS)]
    return (hits[-1] if last else hits[0]) if hits else None


def segment(text, parts, i, stop=""):
    """The stretch of text about part i: from where its key matches to where the next part's key
    (or stop, e.g. where the total starts) does. If no part is named but there's one line per
    part, the lines count in order ("Au / Ag / Fe / Na"). None if part i isn't mentioned."""
    starts = {j: find(p["key"], text) for j, p in enumerate(parts)}
    if starts[i] is None:
        lines = [l for l in text.splitlines() if l.strip()]
        if all(x is None for x in starts.values()) and len(lines) == len(parts):
            return lines[i]
        return None
    later = [x for j, x in starts.items() if j != i and x is not None and x > starts[i]]
    end = find(stop, text[starts[i] + 1:]) if stop else None
    later += [starts[i] + 1 + end] if end else []
    return text[starts[i]:min(later, default=len(text))]


def check_part(part, text):
    want = part["expect"]
    if "number" in want:  # also right if it states the number anywhere ("0.15 × 80 = 12")
        ok, why = check_number(text, float(want["number"]), part.get("q", ""))
        stated = [n for m in re.finditer(r"(?:\bis\b|=|\bequals\b|:|→)\W{0,3}([\d,]+(?:\.\d+)?)", text.lower())
                  for n in numbers(m.group(1))]
        return ok or float(want["number"]) in stated, why
    kind, v = next(iter(want.items()))
    return CHECKS[kind](text, v, {"prompt": part.get("q", "")})


def pick_number(text, prompt):
    """The number an answer gives, by the same rules check_number accepts: after "is" or "=",
    else the first one not in the question, else the last."""
    nums = numbers(text)
    stated = re.search(r"(?:\bis\b|=|\bequals\b)\W{0,3}([\d,]+(?:\.\d+)?|[a-z]+)", text.lower())
    said = numbers(stated.group(1)) if stated else []
    new = [n for n in nums if n not in set(numbers(prompt))]
    return (said or new or nums or [None])[0]


def plain_math(text):
    """LaTeX and markdown to plain text: "$15\\%$ of $80$" → "15% of 80", "\\boxed{415}" → "415"."""
    for a, b in ((r"\\boxed\{([^}]*)\}", r"\1"), (r"\\text\{([^}]*)\}", r"\1"), (r"\\times", "×"),
                 (r"\\div", "÷"), (r"\\%", "%"), (r"\\[\[\]()]|\$|\*\*", "")):
        text = re.sub(a, b, text)
    return text


def score_final(case, answer):
    """A final answer passes when every part is right where it mentions that part, and the
    total (if the case has one) is right where it gives the total. An answer that ends with
    one line per part, in order ("Canberra / Ottawa / ..."), also counts line by line."""
    answer = plain_math(answer)
    parts, fails = case["parts"], []
    tail = [l for l in answer.splitlines() if l.strip()][-len(parts):] if parts else []
    stop = case.get("total", {}).get("key", "")
    for i, part in enumerate(parts):
        # an answer may name a part more than once ("the task asked about banana, apple...");
        # the part is right if it's right where any of those mentions starts
        segs = [seg for at in ([m.start() for m in re.finditer(part["key"], answer, FLAGS)] or [0])
                if (seg := segment(answer[at:], parts, i, stop)) is not None]
        segs += [tail[i]] if len(tail) == len(parts) else []
        checks = [check_part(part, seg) for seg in segs]
        ok, why = next((c for c in checks if c[0]), checks[0] if checks else (False, "not mentioned"))
        if not ok:
            fails.append(f"{part['name']}: {why}")
    if "total" in case:
        t = case["total"]
        ats = [m.start() for m in re.finditer(t["key"], answer, FLAGS)] if t["key"] else [0]
        checks = [check_number(answer[at:], float(t["number"]), case["prompt"]) for at in reversed(ats)]
        ok, why = next((c for c in checks if c[0]), checks[0] if checks else (False, "no total"))
        if not ok:
            fails.append(f"total: {why}")
    return not fails, fails


def score_orchestrated(case, answer, subtasks):
    """Score each stage of an orchestrated run. Each part goes to the first subtask its key
    matches; a worker who got several parts is scored on the stretch about each one."""
    parts = case["parts"]
    owner = [next((k for k, s in enumerate(subtasks) if find(p["key"], s["task"]) is not None), None)
             for p in parts]
    worker_ok, leaked, code_lines, total = [], 0, [], 0.0
    for i, part in enumerate(parts):
        k = owner[i]
        if k is None:
            worker_ok.append(None)  # no subtask asked for it
            code_lines.append(f"{part['name']}: (no subtask)")
            total = None
            continue
        leaked += check_part(part, subtasks[k]["task"])[0]  # the orchestrator wrote the answer into the subtask
        text = plain_math(subtasks[k]["answer"])
        shared = [j for j in range(len(parts)) if owner[j] == k]
        seg = segment(text, [parts[j] for j in shared], shared.index(i)) if len(shared) > 1 else text
        seg = seg if seg is not None else text
        worker_ok.append(check_part(part, seg)[0])
        code_lines.append(f"{part['name']}: {' '.join(seg.split())}")
        n = pick_number(seg, part.get("q", subtasks[k]["task"]))
        total = None if total is None or n is None else total + n
    code = None
    if parts:  # a task that can't be split has nothing for code to merge
        if "total" in case:
            code_lines.append(f"total: {total:g}" if total is not None else "total: (missing a part)")
        code = "\n".join(code_lines)
    merged_ok, fails = score_final(case, answer)
    return {"split": sum(k is not None for k in owner), "parts": len(parts), "n_subtasks": len(subtasks),
            "workers": sum(bool(w) for w in worker_ok), "worker_ok": worker_ok, "leaked": leaked,
            "pass": merged_ok, "fails": fails,
            "code_answer": code, "code_pass": score_final(case, code)[0] if code is not None else None}


def run_orchestrator_suite(suite, cases, chain):
    workers = list(dict.fromkeys(chain[1:] or chain[:1]))
    cfgs = [f"{m} alone" for m in dict.fromkeys([chain[0], *workers])] + ["orchestrated"]
    total, n, results = len(cases) * len(cfgs), 0, []
    start = time.time()
    print(f"{suite['name']}: {len(cases)} cases × {len(cfgs)} configs = {total} runs\n")
    for cfg in cfgs:
        for case in cases:
            n += 1
            t0 = time.time()
            row = {"case": case["id"], "config": cfg}
            try:
                if cfg == "orchestrated":
                    r = bench.run(case["prompt"], chain, "orchestrator", save=False)
                    row.update(score_orchestrated(case, r["answer"], r["subtasks"]),
                               answer=r["answer"], subtasks=r["subtasks"], planned=r["planned"])
                else:
                    r = bench.run(case["prompt"], [cfg.removesuffix(" alone")], "blind", save=False)
                    ok, fails = score_final(case, r["trace"][0]["sent"])
                    row.update({"pass": ok, "fails": fails, "answer": r["trace"][0]["sent"]})
                row["secs"] = r["secs"]
            except Exception as e:  # a crash fails the case; keep going
                row.update({"pass": False, "fails": [f"error: {e}"], "answer": None,
                            "secs": round(time.time() - t0, 1)})
            results.append(row)
            short = " ".join(str(row["answer"] or "").split())[:50]
            extra = (f" split {row['split']}/{row['parts']} workers {row['workers']}/{row['parts']}"
                     f" code {'—' if row['code_pass'] is None else '✓' if row['code_pass'] else '✗'}"
                     if "split" in row else "")
            print(f"[{n:>3}/{total}] {cfg:<24} {case['id']:<12} {'✓' if row['pass'] else '✗'} "
                  f"{row['secs']:>5}s{extra}  {short}")
    return {"suite": suite["name"], "kind": "orchestrator", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prompts": {"plan": bench.PLAN_PROMPT, "merge": bench.MERGE_PROMPT},
            "chain": chain, "configs": cfgs, "cases": [c["id"] for c in cases],
            "secs": round(time.time() - start, 1), "results": results}


def report_orchestrator(rec):
    res = rec["results"]
    print(f"\nSCORECARD  {rec['suite']} · {len(rec['cases'])} cases · {rec['secs'] / 60:.1f} min · "
          f"orchestrator {rec['chain'][0]}, workers {', '.join(rec['chain'][1:]) or rec['chain'][0]}\n")
    print(f"{'config':<26} {'correct':>9} {'avg time':>9}")
    for name in rec["configs"]:
        rows = [r for r in res if r["config"] == name]
        right = sum(r["pass"] for r in rows)
        avg = sum(r["secs"] for r in rows) / max(len(rows), 1)
        print(f"{name:<26} {right:>3}/{len(rows):<3} {right / max(len(rows), 1):>4.0%} {avg:>6.1f}s")
    orch = [r for r in res if r["config"] == "orchestrated" and "split" in r]
    if not orch:
        return
    code = [r for r in orch if r["code_pass"] is not None]
    print(f"{'  merged by code instead':<26} {sum(r['code_pass'] for r in code):>3}/{len(code):<3}"
          f"           (same worker answers, cases that split)")
    parts = sum(r["parts"] for r in orch)
    print(f"\nSTAGES  split: {sum(r['split'] for r in orch)}/{parts} parts got a subtask · "
          f"workers: {sum(r['workers'] for r in orch)}/{parts} parts right · "
          f"leaked: {sum(r.get('leaked', 0) for r in orch)}/{parts} subtasks already held the answer\n")
    print(f"{'case':<13}{'subtasks':>9}{'split':>7}{'workers':>9}{'leaked':>8}  {'orchestrator merge':<20}code merge")
    for r in orch:
        cm = "—" if r["code_pass"] is None else "✓" if r["code_pass"] else "✗"
        print(f"{r['case']:<13}{r['n_subtasks']:>9}{r['split']:>4}/{r['parts']:<2}{r['workers']:>6}/{r['parts']:<2}"
              f"{r.get('leaked', 0):>5}/{r['parts']:<2}  {'✓' if r['pass'] else '✗':<20}{cm}")


# ---- configs and runs ----

def configs(chain):
    return [{"name": f"{m} alone", "chain": [m]} for m in bench.numbered(chain)]


def run_case(case, cfg):
    if case.get("mode") == "cipher":
        return bench.run(case["prompt"], cfg["chain"], "cipher", case.get("rule", ""), save=False,
                         cipher_name=case.get("cipher", "base64"))
    r = bench.run(case["prompt"], cfg["chain"], "blind", case.get("rule", ""), save=False)
    r["answer"] = r["trace"][0]["sent"]
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("suite", nargs="?", default="basics", help="suite name in suites/ or a path")
    p.add_argument("--chain", help=f"models to test, comma-separated (default {bench.DEFAULT_CHAIN}; "
                                   f"orchestrator suite: the orchestrator, then the workers, default {ORCH_CHAIN})")
    p.add_argument("--only", help="comma-separated case ids to run")
    p.add_argument("--show", metavar="FILE", help="print the scorecard of a saved harness run and exit")
    p.add_argument("--rescore", metavar="FILE",
                   help="re-check a saved run's answers with the current checks, without rerunning models")
    a = p.parse_args()

    if a.show:
        rec = json.loads(Path(a.show).read_text())
        (report_orchestrator if rec.get("kind") == "orchestrator" else report)(rec)
        return
    if a.rescore:
        rescore(Path(a.rescore))
        return

    path = Path(a.suite) if a.suite.endswith(".json") else SUITES / f"{a.suite}.json"
    suite = json.loads(path.read_text())
    cases = suite["cases"]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in keep]
    if suite.get("kind") == "orchestrator":
        chain = [m.strip() for m in (a.chain or ORCH_CHAIN).split(",") if m.strip()]
        save(run_orchestrator_suite(suite, cases, chain), report_orchestrator)
        return
    chain = [m.strip() for m in (a.chain or bench.DEFAULT_CHAIN).split(",") if m.strip()]
    cfgs = configs(chain)

    total, n, results = len(cases) * len(cfgs), 0, []
    start = time.time()
    print(f"{suite['name']}: {len(cases)} cases × {len(cfgs)} configs = {total} runs\n")
    for cfg in cfgs:  # one config at a time, so the same models stay loaded
        for case in cases:
            n += 1
            t0 = time.time()
            try:
                r = run_case(case, cfg)
                ok, fails = score(case, r["answer"] or "", r)
                row = {"case": case["id"], "config": cfg["name"], "pass": ok, "fails": fails,
                       "answer": r["answer"], "intact": r.get("intact"), "secs": r["secs"],
                       "tags": case.get("tags", [])}
            except Exception as e:  # a crash fails the case; keep going
                row = {"case": case["id"], "config": cfg["name"], "pass": False, "fails": [f"error: {e}"],
                       "answer": None, "secs": round(time.time() - t0, 1), "tags": case.get("tags", [])}
            results.append(row)
            short = " ".join(str(row["answer"] or "").split())[:50]
            print(f"[{n:>3}/{total}] {cfg['name']:<24} {case['id']:<13} {'✓' if row['pass'] else '✗'} "
                  f"{row['secs']:>5}s  {short}")

    record = {"suite": suite["name"], "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "chain": chain,
              "configs": [c["name"] for c in cfgs], "cases": [c["id"] for c in cases],
              "secs": round(time.time() - start, 1), "results": results}
    save(record, report)


def save(record, show):
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{time.strftime('%Y-%m-%d_%H%M%S')}_{record['suite']}.json"
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    show(record)
    print(f"\nSAVED   {out.relative_to(Path(__file__).parent)}")


def rescore(path):
    rec = json.loads(path.read_text())
    cases = {c["id"]: c for c in json.loads((SUITES / f"{rec['suite']}.json").read_text())["cases"]}
    changed = 0
    if rec.get("kind") == "orchestrator":
        for row in rec["results"]:
            case = cases.get(row["case"])
            if not case or row["answer"] is None:
                continue
            was = (row["pass"], row.get("code_pass"))
            if "subtasks" in row:
                row.update(score_orchestrated(case, row["answer"], row["subtasks"]))
            else:
                row["pass"], row["fails"] = score_final(case, row["answer"])
            changed += was != (row["pass"], row.get("code_pass"))
        path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        report_orchestrator(rec)
        print(f"\nRESCORED {changed} results changed · saved to {path.name}")
        return
    for row in rec["results"]:
        case = cases.get(row["case"])
        if not case or row["answer"] is None or ("intact" in case["check"] and "intact" not in row):
            continue  # unknown case, a crash, or an old cipher row whose result can't be rechecked
        ok, fails = score(case, row["answer"], {"intact": row.get("intact")})
        changed += ok != row["pass"]
        row["pass"], row["fails"] = ok, fails
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    report(rec)
    print(f"\nRESCORED {changed} results changed · saved to {path.name}")


def report(rec):
    res = rec["results"]
    by = lambda **kw: [r for r in res if all(r[k] == v for k, v in kw.items())]
    print(f"\nSCORECARD  {rec['suite']} · {len(rec['cases'])} cases · {rec['secs'] / 60:.1f} min\n")
    print(f"{'config':<26} {'correct':>9} {'avg time':>9}" +
          ("  saboteur" if any(r.get("sabotage") for r in res) else ""))  # saved Relay-era runs had one
    for name in rec["configs"]:
        rows = by(config=name)
        right = sum(r["pass"] for r in rows)
        sab = [r["sabotage"] for r in rows if r.get("sabotage")]
        sab_txt = (f"caught {sab.count('caught')}, slipped {sab.count('slipped')}, "
                   f"didn't lie {sab.count('no_lie')}") if sab else ""
        avg = sum(r["secs"] for r in rows) / max(len(rows), 1)
        print(f"{name:<26} {right:>3}/{len(rows):<3} {right / max(len(rows), 1):>4.0%} {avg:>6.1f}s  {sab_txt}")

    # every case against every config
    cols = rec["configs"]
    print("\n" + " " * 14 + "  ".join(f"{i + 1:>2}" for i in range(len(cols))))
    for cid in rec["cases"]:
        marks = []
        for name in cols:
            r = by(case=cid, config=name)
            marks.append(" ✓" if r and r[0]["pass"] else " ✗")
        print(f"{cid:<14}" + "  ".join(marks))
    print("\n" + "\n".join(f"{i + 1:>2} = {name}" for i, name in enumerate(cols)))


if __name__ == "__main__":
    sys.exit(main())

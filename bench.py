#!/usr/bin/env python3
"""Run a task through local Ollama models and record everything they say.

Modes: blind (every model answers on its own), orchestrator (the first model
splits the task, the rest do the subtasks, the first merges them), vision (models
read a photo) and debate ("roundtable" in code: models answer, debate, and a
moderator or a vote settles it). server.py runs them all from a web page;
harness.py scores them against questions with known answers.

Run:   python3 bench.py "How many r's are in strawberry?"      (blind, default chain)
       python3 bench.py --chain gemma3:1b,gemma3:1b,qwen3:8b "2+2"   (repeats are fine)
       python3 bench.py --json "2+2"     (full trace as JSON)
       python3 bench.py --orchestrator --chain granite4.2:8b,gemma3:1b "Count the r's in strawberry and in mirror"
       python3 bench.py --rule "Answer in one word." "Capital of Japan?"
       python3 bench.py --temperature default "Tell me a fun fact"   (each model's own temperature)
"""

import argparse
import base64
import contextvars
import difflib
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA = "http://localhost:11434"
RUNS = Path(__file__).with_name("runs")  # every run is saved here as one JSON file
DEFAULT_CHAIN = "granite4.2:8b,gemma3:4b,granite4.2:3b,gemma3:1b"
NUM_CTX = 2048      # small context keeps each model's memory down
KEEP_ALIVE = "10m"  # keep models loaded between calls, so repeats and turns don't reload them
MAX_TOKENS = 200    # stops a small model that gets stuck repeating itself
BLIND_TOKENS = 512  # blind answers are the output itself, so give them more room
VISION_TOKENS = 1024  # a page of extracted text can be long
VISION_CTX = 4096     # an image takes a few hundred tokens of context on its own
COMBINE_CTX = 8192    # the final agent reads every extraction at once
MAX_CTX = 32768       # how far a request's context may grow when a big image needs it
THINKS_ALWAYS = ("deepseek-r1",)  # models that reason out loud even when asked not to: more room to answer
OCR_ONLY = ("deepseek-ocr",)      # models that only follow their trained OCR command

SCHEMAS = {
    "bottom": {"type": "object", "properties": {"answer": {"type": "string"}},
               "required": ["answer"]},
    "extract": {"type": "object",
                "properties": {"text": {"type": "string"},
                               "fields": {"type": "array", "items": {
                                   "type": "object",
                                   "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
                                   "required": ["name", "value"]}}},
                "required": ["text", "fields"]},
}


def post(path, body):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


# Temperature for the run in progress: 0 makes every answer repeatable (the harness default);
# None leaves each model's own default, so the same model can answer differently each time.
TEMPERATURE = contextvars.ContextVar("temperature", default=0)
TEMPERATURES = contextvars.ContextVar("temperatures", default={})  # per chain entry, e.g. {"gemma3:1b #2": 1.2}


def base(model):
    """The Ollama model behind a chain entry: a repeated model is labeled "gemma3:1b #2"."""
    return re.sub(r" #\d+$", "", model)


def numbered(chain):
    """Label repeats so every entry is unique (gemma3:1b, gemma3:1b #2, ...). Labels that are
    already unique, as the page sends them, stay as they are."""
    out = []
    for m in chain:
        k, name = 1, m
        while name in out:
            k += 1
            name = f"{base(m)} #{k}"
        out.append(name)
    return out


def ask(model, prompt, kind, rule="", max_tokens=MAX_TOKENS, images=None, num_ctx=NUM_CTX, schema=None):
    """One call to one model. The bottom model replies in plain text, because
    forcing JSON on a 1B model sends it into loops; the others reply in JSON.
    A rule, if given, goes in as the system prompt. images are base64 strings."""
    temp = TEMPERATURES.get().get(model, TEMPERATURE.get())  # this copy's own setting, else the run's
    model = base(model)
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": num_ctx, "num_predict": max_tokens}}
    if temp is not None:
        body["options"]["temperature"] = temp
    if rule:
        body["system"] = rule
    if images:
        body["images"] = images
    if model.startswith(THINKS_ALWAYS):
        body["options"]["num_predict"] = min(max_tokens * 4, 2048)
    if kind != "bottom":
        body["format"] = schema or SCHEMAS[kind]
    start = time.time()
    for _ in range(3):
        try:
            r = post("/api/generate", body)
            break
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            msg = e.read().decode(errors="replace")
            need = re.search(r"request \((\d+) tokens\) exceeds the available context", msg)
            if need and body["options"]["num_ctx"] < MAX_CTX:  # e.g. an image split into many tiles
                want = int(need.group(1)) + body["options"]["num_predict"] + 256
                body["options"]["num_ctx"] = min(MAX_CTX, 1 << (want - 1).bit_length())
            elif "think" in body:
                body.pop("think")  # models without a thinking mode reject the flag
            else:
                raise urllib.error.HTTPError(e.url, e.code, msg, e.headers, None) from None
    else:
        raise RuntimeError(f"{model} kept rejecting the request")
    if not r.get("response", "").strip() and not r.get("eval_count"):
        r = post("/api/generate", body)  # an empty reply is a hiccup (e.g. memory pressure); try once more
    secs = time.time() - start
    if "<think>" in r["response"]:  # reasoning written into the reply; keep only the answer
        answer = re.sub(r"(?s)<think>.*?</think>", "", r["response"])
        r["response"] = "" if "<think>" in answer else answer  # still open: cut off mid-thought
    if "</think>" in r["response"]:  # reasoning with only the closing tag; the answer comes after it
        r["response"] = r["response"].rsplit("</think>", 1)[1]
    try:
        out = json.loads(r["response"]) if kind != "bottom" else {"answer": r["response"].strip()}
    except ValueError:  # usually JSON cut off at the token cap: keep what the answer field had
        text = r["response"].strip()
        m = re.search(r'"(?:answer|instruction)"\s*:\s*"((?:[^"\\]|\\.)*)', text)
        if m:
            try:
                text = json.loads(f'"{m.group(1).rstrip(chr(92))}"')
            except ValueError:
                text = m.group(1)
        out = {"answer": text, "instruction": text}
    out["cut"] = r.get("done_reason") == "length"  # hit the token cap mid-answer
    out["load"] = round(r.get("load_duration", 0) / 1e9, 1)  # seconds spent loading the model first
    toks = r.get("eval_count", 0)
    tps = toks / (r["eval_duration"] / 1e9) if r.get("eval_duration") else 0
    snapshots.append([n for n, _, _ in loaded()])
    return out, round(secs, 1), round(tps)


snapshots = []  # which models were loaded after each call


def loaded():
    return [(m["name"], m.get("size", 0), m.get("size_vram", 0))
            for m in post_get("/api/ps").get("models", [])]


def post_get(path):
    with urllib.request.urlopen(OLLAMA + path, timeout=10) as r:
        return json.load(r)


def norm(text):
    """Loose comparison: case, surrounding quotes, edge punctuation and extra spaces don't count."""
    return " ".join(str(text or "").strip().strip("\"'`.!").lower().split())


# ---- cipher (only the harness's cipher cases use it: one model decodes a message) ----

def _shift(text, k):
    out = []
    for ch in text:
        for base in ("a", "A"):
            if base <= ch <= chr(ord(base) + 25):
                ch = chr((ord(ch) - ord(base) + k) % 26 + ord(base))
                break
        out.append(ch)
    return "".join(out)


def _b64decode(text):
    return base64.b64decode("".join(text.split()), validate=True).decode("utf-8")


CIPHERS = {  # name: (encode, decode, how it's described to the models)
    "base64": (lambda t: base64.b64encode(t.encode()).decode(), _b64decode, "base64"),
    "rot13": (lambda t: _shift(t, 13), lambda t: _shift(t, 13),
              "ROT13 (every letter is shifted 13 places in the alphabet)"),
    "caesar": (lambda t: _shift(t, 3), lambda t: _shift(t, -3),
               "a Caesar cipher with shift 3 (every letter moves 3 places forward, so A becomes D)"),
    "reverse": (lambda t: t[::-1], lambda t: t[::-1],
                "reversal (the whole message is written backwards, last character first)"),
}
MIX = ["base64", "rot13", "reverse", "caesar"]  # "mix" gives each hop the next one of these

DECODE_PROMPT = (
    "You are layer {n} of {total} in a relay chain of AI models passing a secret message. "
    "The message below was encoded with {desc}. Decode it. "
    "Reply with only the decoded message, nothing else.\n\nEncoded message:\n{text}"
)
ENCODE_PROMPT = (
    "You are layer {n} of {total} in a relay chain of AI models passing a secret message. "
    "Encode the message below with {desc} so the next model can decode it. "
    "Reply with only the encoded message, nothing else.\n\nMessage:\n{text}"
)


def clean(text):
    """Strip the code fences and quotes models like to wrap answers in, and keep only
    the first paragraph. Returns (text, whether a commentary paragraph was dropped)."""
    text = re.sub(r"^```\w*\n?|\n?```$", "", text.strip()).strip()
    first, *rest = re.split(r"\n\s*\n", text, maxsplit=1)
    first = first.strip()
    if len(first) > 1 and first[0] == first[-1] and first[0] in "\"'`":
        first = first[1:-1]
    return first.strip(), bool(rest)


def same_encoding(got, want, name):
    if name == "base64":
        return "".join(got.split()) == "".join(want.split())  # base64 is case-sensitive
    return norm(got) == norm(want)


def cipher(task, chain, rule="", on_event=lambda e: None, name="base64"):
    """Pass a secret message down the chain. Each layer decodes what it got and
    re-encodes it for the next; the last one just decodes. The tool knows every
    cipher, so it checks each step exactly instead of trusting the models."""
    snapshots.clear()
    total = len(chain)
    hops = [MIX[i % len(MIX)] if name == "mix" else name for i in range(total)]
    text = CIPHERS[hops[0]][0](task)
    on_event({"type": "start", "cipher": hops[0], "sent": text})
    trace = []

    def step(entry):
        trace.append(entry)
        on_event({"type": "step", **entry})

    decoded = ""
    for i, model in enumerate(chain):
        enc, dec, desc = CIPHERS[hops[i]]
        on_event({"type": "call", "dir": "decode", "layer": i, "model": model})
        out, secs, tps = ask(model, DECODE_PROMPT.format(n=i + 1, total=total, desc=desc, text=text),
                             "bottom", rule)
        decoded, chatty = clean(out["answer"])
        try:
            want = dec(text)
        except ValueError:
            want = None  # what it got wasn't valid for this cipher, so there's no right answer
        step({"dir": "decode", "layer": i, "model": model, "cipher": hops[i], "received": text,
              "sent": decoded, "want": want, "ok": None if want is None else norm(decoded) == norm(want),
              "intact": norm(decoded) == norm(task), "extra": chatty,
              "secs": secs, "tok_s": tps, "cut": out["cut"]})
        if i == total - 1:
            break

        enc, _, desc = CIPHERS[hops[i + 1]]
        on_event({"type": "call", "dir": "encode", "layer": i, "model": model})
        out, secs, tps = ask(model, ENCODE_PROMPT.format(n=i + 1, total=total, desc=desc, text=decoded),
                             "bottom", rule)
        text, chatty = clean(out["answer"])
        want = enc(decoded)
        step({"dir": "encode", "layer": i, "model": model, "cipher": hops[i + 1], "sent": text,
              "want": want, "ok": same_encoding(text, want, hops[i + 1]), "extra": chatty,
              "secs": secs, "tok_s": tps, "cut": out["cut"]})

    first = next((t for t in trace if t["ok"] is not True), None)
    extra = {"cipher": name, "ciphers": hops, "encoded": CIPHERS[hops[0]][0](task),
             "intact": norm(decoded) == norm(task),
             "broke_at": first and {"layer": first["layer"], "model": first["model"],
                                    "dir": first["dir"], "cipher": first["cipher"]}}
    return decoded, trace, extra


# ---- shared by vision and roundtable ----

def word_match(a, b):
    """How much of the original wording survived, 0 to 1, by matching runs of words."""
    words = lambda t: re.findall(r"\w+", t.lower())
    return round(difflib.SequenceMatcher(None, words(a), words(b)).ratio(), 2)


# ---- vision mode ----

_caps = {}


def capabilities(model):
    """What a model can do, e.g. ["completion", "vision", "tools"]. Asked once per model."""
    model = base(model)
    if model not in _caps:
        _caps[model] = post("/api/show", {"model": model}).get("capabilities", [])
    return _caps[model]


FORMATS = ["text", "markdown", "json", "fields"]
EXTRACT_PROMPTS = {
    "text": "Read this image and write out all the text in it, exactly as written, in reading order. "
            "Keep the line breaks. Don't add anything that isn't in the image. Reply with only the text.",
    "markdown": "Read this image and write out all the text in it as Markdown. Keep the layout: headings "
                "as headings, lists as lists, tables as Markdown tables. Don't add anything that isn't "
                "in the image. Reply with only the Markdown.",
    "json": "Read this image. Put all of its text, in reading order, in \"text\". In \"fields\", list "
            "each labeled value you can see (a date, a total, a name, an address and so on) as a name "
            "and a value. Don't add anything that isn't in the image.",
}
COMBINE_PROMPT = (
    "You are the final agent in a chain of AI models. {k} models each read the same {what} and "
    "extracted the text. Their extractions are below; they may disagree or contain mistakes.\n\n"
    "{extractions}\n\nWrite one best version of the text{how}. Where they disagree, go with "
    "what most of them agree on. Don't add anything none of them found.{keep}"
)
TOGETHER_PROMPTS = {  # every photo in one call; each section is labeled so readings stay apart
    "text": "There are {n} images, numbered in the order given. For each one, write \"Image N:\" on its own "
            "line, then all the text in that image, exactly as written, in reading order. Don't add anything "
            "that isn't in the images. Reply with only the text.",
    "markdown": "There are {n} images, numbered in the order given. For each one, start with the heading "
                "\"## Image N\", then all of its text as Markdown, keeping the layout (headings, lists, tables). "
                "Don't add anything that isn't in the images. Reply with only the Markdown.",
    "json": "There are {n} images, numbered in the order given. In \"text\", give each image's text under a "
            "line \"Image N:\". In \"fields\", list each labeled value you can see, starting each name with "
            "\"Image N\" (for example \"Image 2 total\"). Don't add anything that isn't in the images.",
}
COMBINE_HOW = {"text": ", as plain text. Reply with only the text", "markdown": " as Markdown. Reply with only the Markdown",
               "json": ": all of the text in \"text\", and every labeled value in \"fields\""}
ACT_PROMPT = "Here is the text from {what}:\n\n{text}\n\n{instruction}"
STRATEGIES = ["together", "batch", "stream"]

FIELD_TYPES = {"text": "text", "number": "a number", "date": "a date, as written", "money": "an amount of money, as written",
               "list": "a list, one entry per item"}
FIELDS_PROMPT = (
    "Read this image and fill in each field below from what the image says.\n{fields}\n\n"
    "Copy values exactly as they appear. If a field isn't in the image, leave it as an empty string "
    "(or an empty list). Don't guess or make anything up."
)


def clean_fields(fields):
    """[{name, type}], without blanks or repeats; unknown types become text."""
    out, seen = [], set()
    for f in fields or []:
        name = " ".join(str(f.get("name", "")).split())[:60]
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append({"name": name, "type": f.get("type") if f.get("type") in FIELD_TYPES else "text"})
    return out


def field_schema(fields):
    props = {f["name"]: {"type": "array", "items": {"type": "string"}} if f["type"] == "list" else {"type": "string"}
             for f in fields}
    return {"type": "object", "properties": props, "required": [f["name"] for f in fields]}


def field_key(value, ftype):
    """A value in a form where equal readings compare equal; None when the field is empty."""
    if isinstance(value, list):
        items = sorted(norm(x) for x in value if str(x).strip())
        return tuple(items) or None
    text = str(value or "").strip()
    if not text:
        return None
    if ftype in ("number", "money"):
        digits = re.sub(r"[^\d.\-]", "", text.replace(",", ""))
        try:
            return f"{float(digits):g}"
        except ValueError:
            return norm(text)
    return norm(text)


def vote(fields, readings):
    """readings: {model: {field: value}}. For each field, the value most readers gave."""
    grid = {}
    for f in fields:
        name, ftype = f["name"], f["type"]
        values = {m: r.get(name, "") for m, r in readings.items()}
        keys = {m: field_key(v, ftype) for m, v in values.items()}
        counts = {}
        for k in keys.values():
            if k is not None:
                counts[k] = counts.get(k, 0) + 1
        if counts:
            top = max(counts.values())
            best = next(k for k in keys.values() if counts.get(k) == top)  # first reader's, on a tie
            chosen = next(values[m] for m in values if keys[m] == best)
            tie = sum(c == top for c in counts.values()) > 1
        else:
            best, chosen, top, tie = None, [] if ftype == "list" else "", len(values), False
        grid[name] = {"values": values, "match": {m: keys[m] == best for m in values},
                      "chosen": chosen, "agree": top, "tie": tie, "found": best is not None}
    return grid


def as_text(out, fmt):
    """A model's extraction as one string; JSON comes back pretty-printed."""
    if fmt == "json" and "text" in out:
        return json.dumps({"text": out.get("text", ""), "fields": out.get("fields", [])},
                          indent=2, ensure_ascii=False)
    return str(out.get("answer", "")).strip()


def plain(text):
    """The words of an extraction, without JSON keys, for comparing extractions."""
    try:
        d = json.loads(text)
        return " ".join([d.get("text", "")] + [f"{f.get('name', '')} {f.get('value', '')}"
                                                for f in d.get("fields", [])])
    except (ValueError, AttributeError):
        return text


def save_image(image_b64):
    """Keep the image next to the runs, named by its content, so a saved run can show it again."""
    data = base64.b64decode(image_b64)
    name = hashlib.sha1(data).hexdigest()[:16] + ".jpg"
    (RUNS / "images").mkdir(parents=True, exist_ok=True)
    (RUNS / "images" / name).write_bytes(data)
    return name


def vision(instruction, chain, rule="", on_event=lambda e: None, images=(), fmt="text",
           combiner="granite4.2:8b", strategy="batch", fields=()):
    """Fan out, fan in: every model in the chain that can see images extracts the text on
    its own, the combiner merges the readings, and, given an instruction, acts on the result.

    With several photos, strategy picks the order:
      together  each reader gets every photo in one call; one combine for all of them
      batch     every reader reads every photo, then the combiner does each photo
                (fewest model swaps)
      stream    photo by photo: read it, combine it, then start the next
                (first result sooner; reloads models each photo if they don't fit together)

    With fmt "fields", readers fill in only the given fields and code picks each field's
    value by majority vote, so nothing is added at the combining step."""
    images = [i for i in images if i]
    if not images:
        raise ValueError("vision mode needs at least one image")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}")
    snapshots.clear()
    seeing = [m for m in chain if "vision" in capabilities(m)]
    skipped_why = {m: "can't see images" for m in chain if m not in seeing}
    readers = seeing
    if fmt in ("json", "fields"):  # OCR-only models can't follow these instructions
        readers = [m for m in seeing if not m.startswith(OCR_ONLY)]
        skipped_why.update({m: "only reads plain text" for m in seeing if m not in readers})
    if not readers:
        raise ValueError("none of the models in the chain can read the photo this way; "
                         "add gemma3:4b, a gemma4 model or minicpm")
    n, k = len(images), len(readers)
    fields = clean_fields(fields)
    if fmt == "fields" and not fields:
        raise ValueError("add at least one field to fill in")
    if n == 1 or (fmt == "fields" and strategy == "together"):
        strategy = "batch"  # one photo: the orders are the same; fields are filled in per photo
    kind = {"json": "extract", "fields": "fields"}.get(fmt, "bottom")
    schema = field_schema(fields) if fmt == "fields" else None
    grids = {}      # photo -> vote results, for fields
    t0, trace = time.time(), []
    readings = {}   # (reader, photo) -> text; photo is None for "together"
    combined = {}   # photo -> text

    def step(entry):
        entry["at"] = round(time.time() - t0, 1)  # when it finished, from the start of the run
        trace.append(entry)
        on_event({"type": "step", **entry})

    def read(i, photo):
        model = readers[i]
        on_event({"type": "call", "dir": "extract", "layer": i, "model": model, "photo": photo})
        if photo is None:
            prompt, imgs = TOGETHER_PROMPTS[fmt].format(n=n), images
        elif fmt == "fields":
            prompt = FIELDS_PROMPT.format(fields="\n".join(f"- {f['name']} ({FIELD_TYPES[f['type']]})" for f in fields))
            imgs = [images[photo]]
        else:
            prompt, imgs = EXTRACT_PROMPTS[fmt], [images[photo]]
        if model.startswith(OCR_ONLY):
            prompt = "Free OCR."  # its trained command; it ignores longer instructions
        out, secs, tps = ask(model, prompt, kind, rule, VISION_TOKENS * (n if photo is None else 1), imgs,
                             VISION_CTX + (n - 1) * 1024 * (photo is None), schema)
        if fmt == "fields":
            got = {f["name"]: out.get(f["name"], [] if f["type"] == "list" else "") for f in fields}
            readings[i, photo] = json.dumps(got, indent=2, ensure_ascii=False)
        else:
            readings[i, photo] = as_text(out, fmt)
        step({"dir": "extract", "layer": i, "model": model, "photo": photo, "sent": readings[i, photo],
              "format": fmt, "secs": secs, "tok_s": tps, "cut": out["cut"], "load": out["load"]})

    def tally(photo):  # fields: code votes, no model
        on_event({"type": "call", "dir": "vote", "layer": k, "model": "majority vote", "photo": photo})
        grids[photo] = vote(fields, {readers[i]: json.loads(readings[i, photo]) for i in range(k)})
        combined[photo] = json.dumps({f: g["chosen"] for f, g in grids[photo].items()}, indent=2, ensure_ascii=False)
        step({"dir": "vote", "layer": k, "model": "majority vote", "photo": photo, "sent": combined[photo],
              "format": fmt, "grid": grids[photo], "secs": 0, "tok_s": 0})

    def combine(photo):
        if fmt == "fields":
            return tally(photo)
        texts = [readings[i, photo] for i in range(k)]
        extractions = "\n\n".join(f"--- Extraction {j + 1} ({m}) ---\n{t}"
                                   for j, (m, t) in enumerate(zip(readers, texts)))
        what = f"{n} images" if photo is None else "image"
        keep = " Keep the \"Image N\" sections." if photo is None else ""
        on_event({"type": "call", "dir": "combine", "layer": k, "model": combiner, "photo": photo})
        out, secs, tps = ask(combiner, COMBINE_PROMPT.format(k=k, what=what, extractions=extractions,
                                                             how=COMBINE_HOW[fmt], keep=keep),
                             kind, rule, VISION_TOKENS * (n if photo is None else 1), num_ctx=COMBINE_CTX)
        combined[photo] = as_text(out, fmt)
        step({"dir": "combine", "layer": k, "model": combiner, "photo": photo, "sent": combined[photo],
              "format": fmt, "secs": secs, "tok_s": tps, "cut": out["cut"], "load": out["load"]})

    if strategy == "together":
        for i in range(k):
            read(i, None)
        combine(None)
    elif strategy == "batch":
        for i in range(k):          # each reader does every photo while it's loaded
            for photo in range(n):
                read(i, photo)
        for photo in range(n):
            combine(photo)
    else:  # stream
        for photo in range(n):
            for i in range(k):
                read(i, photo)
            combine(photo)

    photos = [None] if strategy == "together" else list(range(n))
    joined = combined[None] if strategy == "together" else (
        combined[0] if n == 1 else "\n\n".join(f"## Photo {p + 1}\n{combined[p]}" for p in photos))
    answer = joined
    act_layer = k + 1 if fmt == "fields" else k  # with fields the vote has its own card
    if instruction:
        on_event({"type": "call", "dir": "act", "layer": act_layer, "model": combiner})
        out, secs, tps = ask(combiner, ACT_PROMPT.format(what="an image" if n == 1 else f"{n} images",
                                                         text=joined, instruction=instruction),
                             "bottom", rule, VISION_TOKENS, num_ctx=COMBINE_CTX)
        answer = out["answer"]
        step({"dir": "act", "layer": act_layer, "model": combiner, "sent": answer, "secs": secs, "tok_s": tps,
              "cut": out["cut"], "load": out["load"]})

    reader_text = lambda i: "\n".join(plain(readings[i, p]) for p in photos)
    names = [save_image(img) for img in images]
    if fmt == "fields":  # agreement: the share of fields where a reader matched the vote
        cells = lambda m: [g["match"][m] for p in photos for g in grids[p].values()]
        agreement = [{"model": m, "match": round(sum(cells(m)) / len(cells(m)), 2)} for m in readers]
    else:
        agreement = [{"model": m, "match": word_match(reader_text(i), "\n".join(plain(combined[p]) for p in photos))}
                     for i, m in enumerate(readers)]
    return answer, trace, {
        "format": fmt, "combiner": combiner, "readers": readers, "combined": joined, "strategy": strategy,
        "skipped": [m for m in chain if m not in readers], "skipped_why": skipped_why,
        "images": names, "image": names[0],
        "first_result": min(t["at"] for t in trace if t["dir"] in ("combine", "vote")),
        "load_secs": round(sum(t.get("load", 0) for t in trace), 1), "agreement": agreement,
        **({"fields": fields, "grids": [grids[p] for p in photos]} if fmt == "fields" else {})}


# ---- roundtable mode ----

ROUNDTABLE_TOKENS = 250  # keep turns short; the transcript grows every turn
ENDINGS = ["vote", "moderator", "both"]
SAY_FIRST = (
    "You are {me}, one of {k} AI models at a roundtable. Here is the question:\n{task}\n\n"
    "Answer it on your own in a few sentences. End with one line that starts with \"My answer:\"."
)
SAY_AGAIN = (
    "You are {me}, one of {k} AI models at a roundtable discussing this question:\n{task}\n\n"
    "The discussion so far:\n{transcript}\n\n"
    "This is discussion round {r} of {rounds}. Read what the others said. Point out anything you think "
    "is wrong, and say whether you've changed your mind and why. Keep it under 120 words. End with two "
    "lines: \"Changed my mind: yes\" or \"Changed my mind: no\", then one that starts with \"My answer:\"."
)
BALLOT_PROMPT = (
    "Question:\n{task}\n\nA roundtable of AI models discussed it. Here are their final answers, "
    "labeled by letter:\n\n{options}\n\nWhich answer is best? Judge correctness first, then clarity. "
    "Give the letter and a one-line reason."
)
MODERATE_PROMPT = (
    "You moderated a roundtable of AI models discussing this question:\n{task}\n\n"
    "The discussion:\n{transcript}\n{votes}\n"
    "Sum up in a few sentences where they agreed and disagreed, then give the best final answer on its "
    "own line, starting with \"Final answer:\"."
)


def tagged(text, tag):
    """What follows the last "<tag>:" in the text, to the end of that line; models often put
    it mid-paragraph rather than on its own line."""
    found = re.findall(rf"(?i){tag}\W*:\**\s*([^\n]+)", text)
    return found[-1].strip().strip("*").strip() if found else None


def my_answer(text):
    """The model's "My answer:", or its last line if it left the tag out."""
    found = tagged(text, "my answer")
    if found is not None:
        return found
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def roundtable(task, chain, rule="", on_event=lambda e: None, rounds=2, ending="vote",
               moderator="granite4.2:8b", allow_self=False):
    """Every model answers alone (round 0), then they discuss over a shared transcript for
    `rounds` rounds; each turn sees everything said so far. It ends with an anonymous
    vote counted by code, a moderator's summary, or both."""
    if len(chain) < 2:
        raise ValueError("a roundtable needs at least two models")
    if ending not in ENDINGS:
        raise ValueError(f"unknown ending {ending!r}")
    snapshots.clear()
    k, rounds = len(chain), max(0, min(int(rounds), 4))
    t0, trace, said = time.time(), [], []  # said: [{model, round, text}]
    answers = {}  # model -> its answer after each round, latest last

    def step(entry):
        entry["at"] = round(time.time() - t0, 1)
        trace.append(entry)
        on_event({"type": "step", **entry})

    def transcript():
        return "\n\n".join(f"[Round {x['round']}] {x['model']}: {x['text']}" for x in said)

    for r in range(rounds + 1):
        for i, model in enumerate(chain):
            on_event({"type": "call", "dir": "say", "layer": i, "model": model, "round": r})
            prompt = (SAY_FIRST.format(me=model, k=k, task=task) if r == 0 else
                      SAY_AGAIN.format(me=model, k=k, task=task, transcript=transcript(), r=r, rounds=rounds))
            out, secs, tps = ask(model, prompt, "bottom", rule, ROUNDTABLE_TOKENS, num_ctx=COMBINE_CTX)
            text, ans = out["answer"], my_answer(out["answer"])
            prev = answers.get(model, [])
            said_so = tagged(text, "changed my mind")  # the model's own word, when it gives it
            if said_so is not None and r > 0:
                changed = said_so.lower().startswith("yes")
            else:  # otherwise guess from how much the answer's wording moved
                changed = bool(prev) and word_match(prev[-1], ans) < 0.5
            answers.setdefault(model, []).append(ans)
            said.append({"model": model, "round": r, "text": text})
            step({"dir": "say", "layer": i, "model": model, "round": r, "sent": text, "answer": ans,
                  "changed": changed, "secs": secs, "tok_s": tps, "cut": out["cut"], "load": out["load"]})

    final = {m: a[-1] for m, a in answers.items()}
    extra = {"rounds": rounds, "ending": ending, "allow_self": allow_self, "answers": final,
             "minds_changed": sum(t["changed"] for t in trace if t["dir"] == "say")}
    answer = None

    if ending in ("vote", "both"):
        tally, ballots = {m: 0 for m in chain}, []
        for i, voter in enumerate(chain):
            options = [m for m in chain if allow_self or m != voter]
            random.Random(f"{task}|{voter}").shuffle(options)  # a different order for every voter
            labels = {chr(65 + j): m for j, m in enumerate(options)}
            text = "\n\n".join(f"{lab}: {final[m]}" for lab, m in labels.items())
            schema = {"type": "object", "properties": {"vote": {"type": "string", "enum": list(labels)},
                                                       "reason": {"type": "string"}}, "required": ["vote", "reason"]}
            on_event({"type": "call", "dir": "ballot", "layer": i, "model": voter})
            out, secs, tps = ask(voter, BALLOT_PROMPT.format(task=task, options=text), "ballot", rule, 120,
                                 num_ctx=COMBINE_CTX, schema=schema)
            pick = labels.get(str(out.get("vote", "")).strip().upper())
            if pick:
                tally[pick] += 1
            ballots.append({"voter": voter, "for": pick, "label": out.get("vote"), "order": labels,
                            "reason": str(out.get("reason", "")).strip(), "spoiled": not pick})
            step({"dir": "ballot", "layer": i, "model": voter, "for": pick, "reason": ballots[-1]["reason"],
                  "spoiled": not pick, "sent": ballots[-1]["reason"], "secs": secs, "tok_s": tps})
        top = max(tally.values())
        winners = [m for m, c in tally.items() if c == top and top > 0]
        extra.update(tally=tally, ballots=ballots, winners=winners)
        answer = final[winners[0]] if len(winners) == 1 else None

    if ending in ("moderator", "both"):
        votes = ""
        if ending == "both":
            votes = "\nThe models then voted: " + ", ".join(f"{m} {c}" for m, c in extra["tally"].items()) + ".\n"
        on_event({"type": "call", "dir": "moderate", "layer": k, "model": moderator})
        out, secs, tps = ask(moderator, MODERATE_PROMPT.format(task=task, transcript=transcript(), votes=votes),
                             "bottom", rule, VISION_TOKENS, num_ctx=COMBINE_CTX)
        extra.update(moderator=moderator, summary=out["answer"])
        answer = tagged(out["answer"], "final answer") or answer or my_answer(out["answer"])
        step({"dir": "moderate", "layer": k, "model": moderator, "sent": out["answer"], "secs": secs,
              "tok_s": tps, "cut": out["cut"], "load": out["load"]})

    if answer is None:  # a tied vote and no moderator
        answer = " / ".join(final[m] for m in extra.get("winners", [])) or "no winner"
    return answer, trace, extra


# ---- orchestrator mode ----

MAX_SUBTASKS = 6
# Prompts say only what the setup needs (who sees what, the subtask limit) and leave the
# thinking to the models: no advice on how to split, check or format. Workers get the
# subtask word for word, as in Blind.
PLAN_PROMPT = (
    "Split this task into subtasks for other AI models to do. Each model sees only its own "
    "subtask. At most {max} subtasks.\n\nTask: {task}"
)
MERGE_PROMPT = (
    "You split this task into subtasks for other AI models:\n{task}\n\n"
    "Their answers:\n{results}\n\nGive the final answer to the task."
)


def orchestrate(task, chain, rule="", on_event=lambda e: None):
    """The first model splits the task into subtasks, the rest of the chain does them (taking turns,
    so one worker model can do them all), and the first model merges the results."""
    snapshots.clear()
    boss, workers = chain[0], chain[1:] or chain[:1]
    trace = []

    def step(entry):
        trace.append(entry)
        on_event({"type": "step", **entry})

    on_event({"type": "call", "dir": "plan", "layer": 0, "model": boss})
    schema = {"type": "object", "required": ["subtasks"], "properties": {"subtasks": {
        "type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": MAX_SUBTASKS}}}
    out, secs, tps = ask(boss, PLAN_PROMPT.format(task=task, max=MAX_SUBTASKS), "plan", rule,
                         BLIND_TOKENS, schema=schema)
    subtasks = [str(t).strip() for t in out.get("subtasks") or [] if str(t).strip()][:MAX_SUBTASKS]
    planned = bool(subtasks)
    subtasks = subtasks or [task]  # an unreadable plan: hand the whole task to one worker
    step({"dir": "plan", "layer": 0, "model": boss, "subtasks": subtasks, "planned": planned,
          "workers": [workers[k % len(workers)] for k in range(len(subtasks))],
          "sent": "\n".join(f"{k + 1}. {t}" for k, t in enumerate(subtasks)),
          "secs": secs, "tok_s": tps, "cut": out["cut"]})

    answers = []
    for k, sub in enumerate(subtasks):
        model = workers[k % len(workers)]
        on_event({"type": "call", "dir": "work", "layer": k + 1, "model": model})
        out, secs, tps = ask(model, sub, "bottom", rule, BLIND_TOKENS)
        answers.append(out["answer"])
        step({"dir": "work", "layer": k + 1, "model": model, "task": sub, "sent": out["answer"],
              "secs": secs, "tok_s": tps, "cut": out["cut"]})

    last = len(subtasks) + 1
    results = "\n\n".join(f"Subtask {k + 1}: {t}\nAnswer: {a}" for k, (t, a) in enumerate(zip(subtasks, answers)))
    on_event({"type": "call", "dir": "merge", "layer": last, "model": boss})
    out, secs, tps = ask(boss, MERGE_PROMPT.format(task=task, results=results), "bottom", rule,
                         VISION_TOKENS, num_ctx=COMBINE_CTX)  # some models work it all out again before merging
    step({"dir": "merge", "layer": last, "model": boss, "sent": out["answer"], "secs": secs, "tok_s": tps,
          "cut": out["cut"]})
    return out["answer"], trace, {"subtasks": [{"task": t, "worker": workers[k % len(workers)], "answer": a}
                                               for k, (t, a) in enumerate(zip(subtasks, answers))],
                                  "planned": planned}


def blind(task, chain, rule="", on_event=lambda e: None):
    """Every model gets the original task word for word and answers on its own.
    No model sees another's answer. One model at a time, so it fits in memory."""
    snapshots.clear()
    trace = []
    for i, model in enumerate(chain):
        on_event({"type": "call", "dir": "solo", "layer": i, "model": model})
        out, secs, tps = ask(model, task, "bottom", rule, BLIND_TOKENS)
        entry = {"dir": "solo", "model": model, "sent": out["answer"], "secs": secs, "tok_s": tps,
                 "cut": out["cut"], "layer": i}
        trace.append(entry)
        on_event({"type": "step", "layer": i, **entry})
    return trace


def run(task, chain, mode="blind", rule="", on_event=lambda e: None, save=True,
        cipher_name="base64", images=(), fmt="text", combiner="granite4.2:8b",
        strategy="batch", fields=(), rounds=2, ending="vote", moderator="granite4.2:8b", allow_self=False,
        temperature=0, temperatures=None):
    """Run one blind, orchestrator, vision or roundtable pass (or a harness cipher case) and return
    its record, saved to runs/ unless save=False. temperature=None uses each model's own default;
    temperatures overrides it for single chain entries ({"gemma3:1b #2": 1.2}; None again means
    the model's own)."""
    temperatures = dict(temperatures or {})
    tokens = TEMPERATURE.set(temperature), TEMPERATURES.set(temperatures)
    try:
        return _run(task, numbered(chain), mode, rule, on_event, save, cipher_name, images, fmt,
                    combiner, strategy, fields, rounds, ending, moderator, allow_self, temperature,
                    temperatures)
    finally:
        TEMPERATURE.reset(tokens[0])
        TEMPERATURES.reset(tokens[1])


def _run(task, chain, mode, rule, on_event, save, cipher_name, images, fmt, combiner, strategy,
         fields, rounds, ending, moderator, allow_self, temperature, temperatures):
    start = time.time()
    if mode == "blind":
        answer, trace, extra = None, blind(task, chain, rule, on_event), {}
    elif mode == "cipher":
        answer, trace, extra = cipher(task, chain, rule, on_event, cipher_name)
    elif mode == "orchestrator":
        answer, trace, extra = orchestrate(task, chain, rule, on_event)
    elif mode == "vision":
        answer, trace, extra = vision(task, chain, rule, on_event, images, fmt, combiner, strategy, fields)
    elif mode == "roundtable":
        answer, trace, extra = roundtable(task, chain, rule, on_event, rounds, ending, moderator, allow_self)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    models = {base(m) for m in chain}
    record = {"mode": mode, "task": task, "rule": rule, "chain": chain, "answer": answer,
              "temperature": temperature,
              **({"temperatures": temperatures} if temperatures else {}), "secs": round(time.time() - start, 1), "calls": len(trace),
              "peak": max(([m for m in snap if m in models] for snap in snapshots), key=len, default=[]),
              **extra, "trace": trace}
    return save_run(record) if save else record


def save_run(record):
    RUNS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    name = record["task"] or f"{record['mode']} {record.get('format', '')}"  # e.g. "vision fields"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40].strip("-") or "run"
    rid, n = f"{stamp}_{slug}", 2
    while (RUNS / f"{rid}.json").exists():
        rid, n = f"{stamp}_{slug}-{n}", n + 1
    record = {"id": rid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
    (RUNS / f"{rid}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return record


def load_run(rid):
    """One saved run by id, or None. Ids are file names, so only plain ones are allowed."""
    path = RUNS / f"{rid}.json"
    if not re.fullmatch(r"[\w-]+", rid) or not path.is_file():
        return None
    return json.loads(path.read_text())


def list_runs():
    """Saved runs, newest first, without their traces."""
    runs = []
    for path in sorted(RUNS.glob("*.json"), reverse=True):
        try:
            r = json.loads(path.read_text())
        except ValueError:
            continue
        runs.append({k: r.get(k) for k in ("id", "time", "mode", "task", "rule", "chain", "answer",
                                           "secs", "calls", "cipher", "intact", "sabotage",
                                           "langs", "match", "format", "combiner", "image", "images",
                                           "readers", "strategy", "rounds", "ending", "winners",
                                           "tally", "minds_changed")})
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("task")
    p.add_argument("--chain", default=DEFAULT_CHAIN, help="models in order, top first, comma-separated")
    p.add_argument("--json", action="store_true", help="print the full trace as JSON")
    p.add_argument("--orchestrator", action="store_true",
                   help="the first model splits the task, the rest do the subtasks, the first merges them "
                        "(without it, every model answers on its own)")
    p.add_argument("--temperature", default="0",
                   help='a number, or "default" for each model\'s own (default 0: repeatable)')
    p.add_argument("--rule", default="", help="a rule every model must follow (system prompt)")
    p.add_argument("--no-save", action="store_true", help="don't save this run to runs/")
    a = p.parse_args()
    chain = [m.strip() for m in a.chain.split(",") if m.strip()]
    mode = "orchestrator" if a.orchestrator else "blind"

    r = run(a.task, chain, mode, a.rule, save=not a.no_save,
            temperature=None if a.temperature == "default" else float(a.temperature))
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return

    arrows = {"solo": "●", "plan": "◇", "work": "●", "merge": "⇢"}
    print(f"TASK  {a.task}\n")
    for t in r["trace"]:
        flag = "  (cut off)" if t.get("cut") else ""
        print(f"{arrows[t['dir']]} {t['model']:<16} {t['secs']:>5}s {t['tok_s']:>4} tok/s{flag}")
        if t["dir"] == "work":
            print(f"    subtask: {t['task']}")
        print("    " + t["sent"].replace("\n", "\n    "))
    print(f"\nPEAK    {len(r['peak'])}/{len(chain)} models loaded at once: {', '.join(r['peak'])}")
    if r["answer"] is not None:
        print(f"ANSWER  {r['answer']}")
    print(f"TIME    {r['secs']}s for {r['calls']} calls")
    if "id" in r:
        print(f"SAVED   runs/{r['id']}.json")


if __name__ == "__main__":
    sys.exit(main())

"""Per-class binary yes/no query support for Gemma 4.

Separate from the ranked zero-shot path in zeroshot.py — the binary path
asks ONE question at a time ("does this image show X?"), so it needs its
own prompt template, parser, and inference loop.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
from tqdm import tqdm


BINARY_PROMPT_TEMPLATE = """Look at the image carefully.

Question: Does this image show {description}

Answer in this EXACT format on one line, nothing else:
{{"answer": "yes", "confidence": 85, "rationale": "short sentence"}}
or
{{"answer": "no", "confidence": 90, "rationale": "short sentence"}}

The confidence is an integer from 0 to 100. The rationale must be under 15 words.
"""


def build_binary_prompt(description: str) -> str:
    """Format one binary question for Gemma.

    The ``description`` should be the human-readable class description
    from client_data.CLIENT_DESCRIPTIONS (without the "xxx —" prefix if
    possible). We drop the class id prefix to focus the model on the
    visual meaning instead of matching a string.
    """
    # strip leading "class_name — " pattern if present
    cleaned = re.sub(r"^[a-z_/]+\s*—\s*", "", description)
    if not cleaned.endswith("?"):
        cleaned = cleaned.rstrip(".") + "?"
    return BINARY_PROMPT_TEMPLATE.format(description=cleaned)


_RESPONSE_RE = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.DOTALL)


def parse_binary_response(text: str) -> dict:
    """Extract {answer, confidence, rationale} from Gemma's output.

    Returns a dict with:
      - answer: "yes" | "no" | None (if unparseable)
      - confidence: int in 0..100, or None
      - rationale: str or empty
      - parse_ok: bool
    """
    # primary: pull the first JSON-looking object
    m = _RESPONSE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            ans = str(obj.get("answer", "")).strip().lower()
            if ans not in ("yes", "no"):
                ans = None
            conf = obj.get("confidence")
            try:
                conf = int(conf)
                if not (0 <= conf <= 100):
                    conf = None
            except (TypeError, ValueError):
                conf = None
            rationale = str(obj.get("rationale", "")).strip()
            return {
                "answer": ans,
                "confidence": conf,
                "rationale": rationale,
                "parse_ok": ans is not None,
            }
        except json.JSONDecodeError:
            pass

    # fallback: look for yes/no token in the raw text
    lowered = text.lower()
    m_yes = re.search(r"\b(yes)\b", lowered)
    m_no = re.search(r"\b(no)\b", lowered)
    ans = None
    if m_yes and m_no:
        ans = "yes" if m_yes.start() < m_no.start() else "no"
    elif m_yes:
        ans = "yes"
    elif m_no:
        ans = "no"
    return {
        "answer": ans,
        "confidence": None,
        "rationale": "",
        "parse_ok": False,
    }


def _apply_record_to_arrays(record: dict, scores: np.ndarray, answers: np.ndarray, parse_ok: np.ndarray) -> None:
    """Fold one JSONL record into the running (N, K) arrays."""
    i, j = int(record["i"]), int(record["j"])
    ans = record.get("answer")
    conf = record.get("confidence")
    parse_ok[i, j] = bool(record.get("parse_ok", False))
    if ans == "yes":
        answers[i, j] = 1
        scores[i, j] = (conf if isinstance(conf, int) else 50) / 100.0
    elif ans == "no":
        answers[i, j] = 0
        scores[i, j] = 1.0 - ((conf if isinstance(conf, int) else 50) / 100.0)
    else:
        answers[i, j] = -1
        scores[i, j] = 0.5


def _load_existing_stream(stream_path: Path, N: int, K: int) -> tuple[set[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Read an existing JSONL stream and rebuild (i, j) done set + arrays.

    Returns (done_set, scores, answers, parse_ok, records). If the file
    doesn't exist, returns an empty done set and zeroed arrays.
    """
    scores = np.zeros((N, K), dtype=np.float32)
    answers = np.full((N, K), -1, dtype=np.int8)
    parse_ok = np.zeros((N, K), dtype=bool)
    done: set[tuple[int, int]] = set()
    records: list[dict] = []
    if not stream_path.exists():
        return done, scores, answers, parse_ok, records
    with stream_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i, j = int(r.get("i", -1)), int(r.get("j", -1))
            if 0 <= i < N and 0 <= j < K:
                if (i, j) in done:
                    continue  # dedupe — if the same pair was retried, keep the first
                done.add((i, j))
                _apply_record_to_arrays(r, scores, answers, parse_ok)
                records.append(r)
    return done, scores, answers, parse_ok, records


def gemma_binary_verify(
    model,
    processor,
    config,
    image_paths: Sequence[str],
    labels: Sequence[int],
    classes: Sequence[str],
    descriptions: dict[str, str],
    max_tokens: int = 60,
    stream_path: str | Path | None = None,
    desc: str = "Gemma-4 binary",
    resume: bool = True,
) -> dict:
    """Run Gemma per-class binary verification across all (image, class) pairs.

    Supports **resume-from-stream**: if ``stream_path`` already exists and
    ``resume=True``, any (i, j) pair already recorded in the file is
    skipped and its result re-applied to the output arrays from the
    stream. The stream is opened in append mode so crashes mid-run don't
    lose prior progress. To force a fresh run, either delete the stream
    file or pass ``resume=False``.

    Returns a dict:
      - scores:     (N, K) float confidence scores in [0, 1];
                    a "no" answer is encoded as (1 - confidence/100),
                    a "yes" as (confidence/100), so the score is always
                    "probability of this class being present"
      - answers:    (N, K) int array: 1 for yes, 0 for no, -1 if parse failed
      - raw:        list of per-call audit records (resumed + new combined)
      - parse_ok:   bool array
    """
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    N, K = len(image_paths), len(classes)
    total = N * K

    # ── load existing stream if any ─────────────────────────────
    done: set[tuple[int, int]] = set()
    scores = np.zeros((N, K), dtype=np.float32)
    answers = np.full((N, K), -1, dtype=np.int8)
    parse_ok = np.zeros((N, K), dtype=bool)
    raw_records: list[dict] = []

    if stream_path is not None:
        stream_path = Path(stream_path)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and stream_path.exists():
            done, scores, answers, parse_ok, raw_records = _load_existing_stream(stream_path, N, K)
            print(f"[resume] loaded {len(done)}/{total} pre-existing calls from {stream_path}", flush=True)

    stream_fh = None
    if stream_path is not None:
        mode = "a" if (resume and stream_path.exists()) else "w"
        stream_fh = stream_path.open(mode)

    try:
        pbar = tqdm(total=total, desc=desc, initial=len(done))
        for i, path in enumerate(image_paths):
            for j, cls in enumerate(classes):
                if (i, j) in done:
                    continue  # already done in a previous run, skip inference

                prompt_text = build_binary_prompt(descriptions[cls])
                formatted = apply_chat_template(processor, config, prompt_text, num_images=1)
                out = generate(
                    model=model,
                    processor=processor,
                    prompt=formatted,
                    image=[path],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                text = out.text if hasattr(out, "text") else str(out)
                parsed = parse_binary_response(text)

                record = {
                    "i": i,
                    "j": j,
                    "path": str(path),
                    "true": classes[int(labels[i])],
                    "query_class": cls,
                    "is_target_class": classes[int(labels[i])] == cls,
                    "answer": parsed["answer"],
                    "confidence": parsed["confidence"],
                    "parse_ok": bool(parsed["parse_ok"]),
                    "rationale": parsed["rationale"][:200],
                    "raw": text.strip()[:400],
                }
                _apply_record_to_arrays(record, scores, answers, parse_ok)

                if stream_fh is not None:
                    stream_fh.write(json.dumps(record) + "\n")
                    stream_fh.flush()
                raw_records.append(record)
                done.add((i, j))
                pbar.update(1)
        pbar.close()
    finally:
        if stream_fh is not None:
            stream_fh.close()

    return {
        "scores": scores,
        "answers": answers,
        "parse_ok": parse_ok,
        "raw": raw_records,
    }

"""LEDGAR F1 harness for :class:`ZeroShotClassify` over a
100-clause-type label space.

Measures: does ZeroShotClassify pick the right contract clause type
when the label set is large (100 classes) and the inputs are real
contract clauses? Reports F1-macro, F1-micro, accuracy, and a
per-class breakdown of the worst- and best-performing classes.

Data source:
    LEDGAR (Tuggener et al. 2020, LREC) is a labelled corpus of
    contract clauses with 100 fine-grained clause types. The LexGLUE
    subset (Chalkidis et al. 2022, ACL) trims it to a clean
    train/dev/test split. **License:** CC-BY-NC-SA-4.0
    (non-commercial, share-alike). The harness downloads at runtime
    via HuggingFace ``datasets``; we do NOT vendor the corpus into
    this repository. Users running this harness must comply with the
    upstream license terms for their use case (the harness output
    JSON is **not** a redistribution of the dataset; it's evaluation
    metadata only).

Run with::

    KAOS_LLM_LIVE_PROVIDER=anthropic \\
        ANTHROPIC_API_KEY=... \\
        uv run --no-sync pytest tests/quality/test_ledgar_f1.py \\
            --no-cov -v -m live -s

Cost (claude-haiku-4-5, 100 clauses, ~200 tokens each, 100-class
LabelSet in prompt): ~$0.50-1.00 per run. Skip cleanly when
``KAOS_LLM_LIVE_PROVIDER`` is unset OR ``datasets`` /
``scikit-learn`` aren't installed.

Output: ``docs/benchmarks/quality-ledgar-f1.json`` with the
per-clause prediction trace + aggregate F1-macro / F1-micro /
accuracy + per-class precision / recall / support.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.live

_BENCH_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"

_DEFAULT_N_CLAUSES = int(os.environ.get("KAOS_LEDGAR_N", "100"))


def _live_provider_available() -> bool:
    return bool(os.environ.get("KAOS_LLM_LIVE_PROVIDER"))


@pytest.fixture(scope="module")
def ledgar_split():
    """Return ``(clauses, labels, label_names)`` from LEDGAR test split.

    ``label_names`` is the canonical 100-class list; ``labels`` are
    integer indices into ``label_names``. Skips cleanly when
    ``datasets`` / ``scikit-learn`` are missing.
    """
    datasets = pytest.importorskip(
        "datasets", reason="LEDGAR harness requires `pip install datasets`"
    )
    pytest.importorskip("sklearn", reason="LEDGAR harness requires `pip install scikit-learn`")
    ds = datasets.load_dataset("lex_glue", "ledgar", split="test")
    label_names: list[str] = list(ds.features["label"].names)

    # Stratified sample: try to hit at least 1 example per of the
    # most common N classes to keep the F1 macro signal real. Falls
    # back to a head-of-split slice when stratification can't be
    # cleanly fulfilled.
    counts: Counter = Counter()
    target_per_class = max(1, _DEFAULT_N_CLAUSES // 30)
    rows: list[tuple[str, int]] = []
    for row in ds:
        lab = int(row["label"])
        if counts[lab] >= target_per_class:
            continue
        counts[lab] += 1
        rows.append((row["text"], lab))
        if len(rows) >= _DEFAULT_N_CLAUSES:
            break
    if len(rows) < _DEFAULT_N_CLAUSES:
        # Top up with head-of-split.
        seen = {id(r) for r in rows}
        for row in ds:
            r = (row["text"], int(row["label"]))
            if id(r) in seen:
                continue
            rows.append(r)
            if len(rows) >= _DEFAULT_N_CLAUSES:
                break

    clauses = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return clauses, labels, label_names


def _make_labelset(names: list[str]):
    from kaos_llm_core.labels import Label, LabelSet

    return LabelSet(
        labels=[
            Label(
                name=name,
                description=f"Contract clause of type '{name.replace('_', ' ')}'.",
            )
            for name in names
        ],
        exclusive=True,
        allow_abstain=True,
    )


async def _classify_one(classifier, clause: str) -> dict:
    result = await classifier(text=clause)
    return {
        "picked": [lab.name for lab in result.labels],
        "abstained": bool(result.abstained),
        "scores": dict(result.scores) if result.scores else {},
    }


@pytest.mark.skipif(
    not _live_provider_available(),
    reason="KAOS_LLM_LIVE_PROVIDER not set; LEDGAR harness requires live LLM access",
)
def test_ledgar_f1(ledgar_split) -> None:
    """100-class ZeroShotClassify F1-macro + F1-micro on LEDGAR test slice."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    from kaos_llm_core.programs.classify import ZeroShotClassify

    clauses, gold_indices, label_names = ledgar_split
    if not clauses:
        pytest.skip("LEDGAR test split returned no usable rows")

    label_set = _make_labelset(label_names)
    classifier = ZeroShotClassify(labels=label_set)

    rows: list[dict] = []
    predictions: list[int | None] = []
    for idx, (clause, gold_idx) in enumerate(zip(clauses, gold_indices, strict=True)):
        try:
            res = asyncio.run(_classify_one(classifier, clause))
        except Exception as exc:
            rows.append(
                {
                    "index": idx,
                    "gold": label_names[gold_idx],
                    "error": str(exc),
                }
            )
            predictions.append(None)
            continue
        picked = res["picked"][0] if res["picked"] else None
        pred_idx = label_names.index(picked) if picked in label_names else None
        rows.append(
            {
                "index": idx,
                "gold": label_names[gold_idx],
                "predicted": picked,
                "abstained": res["abstained"],
                "correct": pred_idx == gold_idx,
            }
        )
        predictions.append(pred_idx)

    # Compute scikit-learn metrics. Skip rows with prediction=None
    # (errors); they're tracked separately.
    paired = [(g, p) for g, p in zip(gold_indices, predictions, strict=True) if p is not None]
    if not paired:
        pytest.fail("Every LEDGAR clause errored -- check provider connectivity / quota")
    gold_arr = [p[0] for p in paired]
    pred_arr = [p[1] for p in paired]

    f1_macro = round(f1_score(gold_arr, pred_arr, average="macro", zero_division=0), 4)
    f1_micro = round(f1_score(gold_arr, pred_arr, average="micro", zero_division=0), 4)
    acc = round(accuracy_score(gold_arr, pred_arr), 4)

    # Per-class report (only emit classes that actually appear).
    seen_classes = sorted(set(gold_arr) | set(pred_arr))
    cls_report = classification_report(
        gold_arr,
        pred_arr,
        labels=seen_classes,
        target_names=[label_names[i] for i in seen_classes],
        zero_division=0,
        output_dict=True,
    )
    per_class: list[dict] = []
    for lab in seen_classes:
        name = label_names[lab]
        stats = cls_report.get(name, {})
        if not isinstance(stats, dict):
            continue
        per_class.append(
            {
                "label": name,
                "precision": round(float(stats.get("precision", 0.0)), 4),
                "recall": round(float(stats.get("recall", 0.0)), 4),
                "f1": round(float(stats.get("f1-score", 0.0)), 4),
                "support": int(stats.get("support", 0)),
            }
        )

    n_errors = sum(1 for r in rows if "error" in r)
    aggregate = {
        "n_clauses": len(clauses),
        "n_scored": len(paired),
        "n_errors": n_errors,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy": acc,
        "n_classes_seen": len(seen_classes),
    }

    payload = {
        "harness": "LEDGAR ZeroShotClassify F1",
        "dataset": "lex_glue/ledgar test split (CC-BY-NC-SA-4.0 -- not vendored)",
        "n_label_names": len(label_names),
        "captured_at": "2026-05-15",
        "aggregate": aggregate,
        "per_class": per_class,
        "per_cell": rows,
    }

    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _BENCH_DIR / "quality-ledgar-f1.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")

    if os.environ.get("KAOS_LLM_BENCH_PRINT"):
        print("\nLEDGAR aggregate:", aggregate)

    # Soft assertion: ZeroShotClassify on 100 classes should beat
    # uniform random (F1 ~1/100 = 0.01) by a wide margin. We expect
    # claude-haiku-4-5 to land somewhere in the 0.30-0.55 F1-macro
    # range on this slice based on the literature.
    assert acc > 0.05, (
        f"LEDGAR accuracy {acc} is barely above random; inspect "
        f"per-class report for systematic class confusion."
    )

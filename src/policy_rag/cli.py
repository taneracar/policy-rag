"""Command line: build the index, ask a question, run the eval.

policy-rag ask "How many sick days do we get?"
policy-rag eval --out reports/eval.json
policy-rag eval --answerer claude          # real model, spends API credits
policy-rag eval --store postgres --dsn postgresql://...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .answer import ClaudeAnswerer, ExtractiveAnswerer
from .chunking import load_corpus
from .embed import HashEmbedder, Model2VecEmbedder
from .evaluate import (
    calibrate,
    check_grounding_rejects_fabrication,
    compare_modes,
    end_to_end,
    load_golden,
    split,
    summarize,
)
from .graph import ask, build_graph
from .retrieve import MODES, Retriever
from .store import PgVectorStore, SourceFilter, SqliteStore

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus"
GOLDEN = ROOT / "eval" / "golden.jsonl"
DEFAULTS = ROOT / "eval" / "calibration.json"


def make_embedder(name: str):
    return HashEmbedder() if name == "hash" else Model2VecEmbedder()


def make_store(args, dim: int):
    if args.store == "postgres":
        dsn = args.dsn or os.environ.get("POLICY_RAG_PG_DSN")
        if not dsn:
            sys.exit("--store postgres needs --dsn or POLICY_RAG_PG_DSN")
        return PgVectorStore(dsn, dim, reset=True)
    return SqliteStore(args.db)


def build(args):
    embedder = make_embedder(args.embedder)
    store = make_store(args, embedder.dim)
    chunks = load_corpus(Path(args.corpus))
    store.add(chunks, embedder.embed([c.embedding_text() for c in chunks]))
    return store, embedder


def source_filter(args) -> SourceFilter:
    return SourceFilter(args.include_superseded, args.include_unreviewed)


def make_answerer(args, embedder, min_score: float):
    if args.answerer == "claude":
        return ClaudeAnswerer(effort=args.effort)
    return ExtractiveAnswerer(embedder, min_score=min_score)


def load_defaults() -> dict:
    return json.loads(DEFAULTS.read_text()) if DEFAULTS.exists() else {"min_cosine": 0.0, "min_score": 0.45}


def cmd_ask(args) -> None:
    store, embedder = build(args)
    d = load_defaults()
    graph = build_graph(
        Retriever(store, embedder, args.mode, source_filter=source_filter(args)),
        make_answerer(args, embedder, d["min_score"]),
        args.k,
        d["min_cosine"],
    )
    print(json.dumps(ask(graph, args.question).to_dict(), indent=2))


def cmd_eval(args) -> None:
    store, embedder = build(args)
    cases = load_golden(Path(args.golden))
    dev, test = split(cases, "dev"), split(cases, "test")

    retrieval = {
        name: compare_modes(lambda m: Retriever(store, embedder, m, source_filter=source_filter(args)), part)
        for name, part in [("dev", dev), ("test", test)]
    }

    def make_graph(min_cosine: float, min_score: float):
        return build_graph(
            Retriever(store, embedder, args.mode, source_filter=source_filter(args)),
            make_answerer(args, embedder, min_score),
            args.k,
            min_cosine,
        )

    if args.answerer == "claude" or args.no_calibrate:
        d = load_defaults()
        cal = {"min_cosine": d["min_cosine"], "min_score": d["min_score"], "source": str(DEFAULTS.name)}
    else:
        c = calibrate(
            make_graph, dev, cosines=[0.0, 0.2, 0.25, 0.3, 0.35], scores=[0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
        )
        cal = {
            "min_cosine": c.min_cosine,
            "min_score": c.min_score,
            "dev_balanced_accuracy": c.dev_balanced_accuracy,
            "source": "dev grid",
        }

    graph = make_graph(cal["min_cosine"], cal["min_score"])
    rows = {name: end_to_end(graph, part) for name, part in [("dev", dev), ("test", test)]}
    report = {
        "corpus": "synthetic (fictional company, written for this demo)",
        "store": store.name,
        "embedder": embedder.name,
        "answerer": graph_answerer_name(args, embedder),
        "mode": args.mode,
        "k": args.k,
        "chunks": store.count(),
        "source_filter": vars(source_filter(args)),
        "calibration": cal,
        "retrieval": retrieval,
        "end_to_end": {name: summarize(r) for name, r in rows.items()},
        "forged_number_claims_accepted": check_grounding_rejects_fabrication(rows["dev"] + rows["test"]),
        "failures": [
            {"id": r.case.id, "split": r.case.split, **r.result.to_dict()}
            for part in rows.values()
            for r in part
            if not r.correct
        ],
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(render_markdown(report))
    if args.save_calibration and args.answerer != "claude":
        DEFAULTS.write_text(
            json.dumps({"min_cosine": cal["min_cosine"], "min_score": cal["min_score"]}, indent=2) + "\n"
        )
    if args.gate:
        failed = gate(report, json.loads(Path(args.gate).read_text()))
        for f in failed:
            print(f"GATE FAILED: {f}", file=sys.stderr)
        if failed:
            sys.exit(1)


def graph_answerer_name(args, embedder) -> str:
    return (
        ClaudeAnswerer.__name__ + f" ({args.effort})"
        if args.answerer == "claude"
        else ExtractiveAnswerer(embedder).name
    )


def gate(report: dict, thresholds: dict) -> list[str]:
    """thresholds: {"retrieval.test.hybrid.recall@5": 0.9, "end_to_end.test.refusal_accuracy": 0.8, ...} (all minimums)."""
    failed = []
    for path, minimum in thresholds.items():
        if path.startswith("max:"):
            path, is_max = path[4:], True
        else:
            is_max = False
        value = report
        for key in path.split("."):
            value = value[key]
        if value is None or (value > minimum if is_max else value < minimum):
            failed.append(f"{path} = {value} ({'max' if is_max else 'min'} {minimum})")
    return failed


def render_markdown(r: dict) -> str:
    lines = [
        f"Corpus: {r['corpus']}, {r['chunks']} chunks. Store: {r['store']}. Embedder: {r['embedder']}.",
        f"Answerer: {r['answerer']}. Retrieval for answers: {r['mode']}, k={r['k']}. Source filter: {r['source_filter']}.",
        f"Thresholds from {r['calibration']['source']}: gate cosine >= {r['calibration']['min_cosine']}, "
        f"answerer min score {r['calibration']['min_score']}.",
        "",
        "Retrieval (answerable questions)",
        "",
        "| split | mode | recall@1 | recall@3 | recall@5 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for sp, modes in r["retrieval"].items():
        for mode in MODES:
            m = modes[mode]
            lines.append(
                f"| {sp} (n={m['n']}) | {mode} | {m['recall@1']:.2f} | {m['recall@3']:.2f} | {m['recall@5']:.2f} | {m['mrr']:.2f} |"
            )
    lines += [
        "",
        "End to end",
        "",
        "| split | n | answer acc. | refusal acc. | false refusals | groundedness | citation acc. | model calls | cost/query |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sp, s in r["end_to_end"].items():
        cost = f"${s['cost_per_query_usd']:.4f}" + (" (projected)" if s["cost_is_projection"] else "")
        lines.append(
            f"| {sp} | {s['n']} | {_pct(s['answer_accuracy'])} | {_pct(s['refusal_accuracy'])} | {_pct(s['false_refusal_rate'])} "
            f"| {_pct(s['groundedness'])} | {_pct(s['citation_accuracy'])} | {s['model_calls']} | {cost} |"
        )
    lines += ["", "Accuracy by question kind (all splits)", ""]
    kinds: dict[str, list] = {}
    for sp, s in r["end_to_end"].items():
        for kind, v in s["accuracy_by_kind"].items():
            kinds.setdefault(kind, []).append(f"{sp} {_pct(v)}")
    lines += [f"- {kind}: {', '.join(v)}" for kind, v in sorted(kinds.items())]
    lines += ["", f"Claims with a forged number that passed the grounding check: {r['forged_number_claims_accepted']}"]
    lines += ["", f"Failures: {len(r['failures'])}"]
    for f in r["failures"]:
        got = f["answer"] or f"refused ({f['refusal_reason']})"
        lines.append(f"- {f['id']} [{f['split']}] {f['question']} -> {got[:110]}")
    return "\n".join(lines)


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="policy-rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", default=str(CORPUS))
    p.add_argument("--store", choices=["sqlite", "postgres"], default="sqlite")
    p.add_argument("--db", default=":memory:", help="SQLite path (default: in memory)")
    p.add_argument("--dsn", help="Postgres DSN for --store postgres")
    p.add_argument("--embedder", choices=["model2vec", "hash"], default="model2vec")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--include-superseded", action="store_true", help="also search superseded policy versions")
    p.add_argument("--include-unreviewed", action="store_true", help="also search sources no policy owner reviewed")
    p.add_argument("--answerer", choices=["extractive", "claude"], default="extractive")
    p.add_argument("--effort", default="low", help="Claude effort level (with --answerer claude)")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask")
    a.add_argument("question")
    e = sub.add_parser("eval")
    e.add_argument("--golden", default=str(GOLDEN))
    e.add_argument("--out")
    e.add_argument("--gate", help="JSON file of metric thresholds; exit 1 if any is missed")
    e.add_argument("--no-calibrate", action="store_true", help="use eval/calibration.json instead of tuning on dev")
    e.add_argument("--save-calibration", action="store_true")
    args = p.parse_args(argv)
    {"ask": cmd_ask, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()

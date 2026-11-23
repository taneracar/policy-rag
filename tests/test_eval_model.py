"""End-to-end eval with the real local embedding model (downloads it once). Run: pytest -m model"""

import json

import pytest

from policy_rag.cli import ROOT, gate, main

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    out = tmp_path_factory.mktemp("reports")
    main(["eval", "--out", str(out / "default.json")])
    main(
        ["--include-superseded", "--include-unreviewed", "eval", "--no-calibrate", "--out", str(out / "nofilter.json")]
    )
    return json.loads((out / "default.json").read_text()), json.loads((out / "nofilter.json").read_text())


def test_default_run_meets_the_gate(reports):
    default, _ = reports
    assert gate(default, json.loads((ROOT / "eval" / "gate.json").read_text())) == []


def test_source_filter_is_what_blocks_the_injection_and_stale_values(reports):
    default, nofilter = reports
    kind = lambda r, sp, k: r["end_to_end"][sp]["accuracy_by_kind"][k]  # noqa: E731
    assert kind(default, "test", "injection-bait") == 1.0
    assert kind(nofilter, "test", "injection-bait") == 0.0
    assert kind(nofilter, "test", "superseded-conflict") < kind(default, "test", "superseded-conflict")


def test_hybrid_is_not_worse_than_either_ranker_on_test_mrr(reports):
    r = reports[0]["retrieval"]["test"]
    assert r["hybrid"]["mrr"] >= max(r["dense"]["mrr"], r["keyword"]["mrr"]) - 1e-9

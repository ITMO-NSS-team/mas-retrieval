import csv
import io
import json

import pytest

from experiments.benchmarks.frames.builder import (
    FramesBuilder, article_id, chunks, convert_questions, digest, freeze_snapshot,
    normalize_url, parse_html, split_ids, write_json, write_jsonl,
)
from marlib.benchmarks.base import BenchmarkSpec


def tsv_row():
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=["", "Prompt", "Answer", "wikipedia_link_1", "wikipedia_link_11+", "wiki_links", "reasoning_types"], delimiter="\t")
    writer.writeheader()
    writer.writerow({"": "7", "Prompt": "Q", "Answer": "A", "wikipedia_link_1": "https://en.wikipedia.org/wiki/Alpha#Section",
                     "wikipedia_link_11+": "['https://en.wikipedia.org/wiki/Quincy,_Massachusetts', 'https://en.wikipedia.org/wiki/Beta']",
                     "wiki_links": "['https://en.wikipedia.org/wiki/Alpha']", "reasoning_types": "Numerical reasoning | Temporal reasoning"})
    return out.getvalue().encode()


def test_complete_url_fields_and_stable_ids():
    q = convert_questions(tsv_row())[0]
    assert q["id"] == "frames_0007"
    assert q["n_sources"] == 3
    assert any("Quincy,_Massachusetts" in u for u in q["normalized_urls"])
    assert len(q["reasoning_types"]) == 2
    assert normalize_url("http://en.m.wikipedia.org/wiki/A%20B#Part") == normalize_url("https://en.wikipedia.org/wiki/A_B")


def test_table_list_and_date_preserved():
    html = b'<title>Title</title><div class="mw-parser-output"><p>As of August 2024.</p><table><tr><th>Year</th><th>Value</th></tr><tr><td>2023</td><td>17</td></tr></table><ul><li>First fact</li><li>Second fact</li></ul><script>hidden</script></div>'
    _, _, text = parse_html(html, "https://en.wikipedia.org/wiki/Alpha")
    assert "[Year | Value] 2023 | 17" in text
    assert "First fact\nSecond fact" in text
    assert "August 2024" in text
    assert "hidden" not in text
    assert max(map(len, chunks("x " * 5000))) <= 2400


def test_protection_badge_does_not_hide_article():
    html = b'<div class="mw-parser-output">badge</div><div id="mw-content-text"><div class="mw-parser-output"><p>Actual article.</p></div></div>'
    _, _, text = parse_html(html, "https://en.wikipedia.org/wiki/Alpha")
    assert text == "Actual article."


def test_category_lists_outside_parser_output():
    html = b'<div id="mw-content-text"><div class="mw-parser-output"></div><div class="mw-category"><ul><li>London 2012</li></ul></div></div>'
    assert "London 2012" in parse_html(html, "https://en.wikipedia.org/wiki/Category:Olympics")[2]


def setup_snapshot(tmp_path, missing=False):
    spec = BenchmarkSpec("frames", tmp_path, "desc", "frames")
    directory = spec.source_dir / "doc_html"
    directory.mkdir(parents=True)
    mapping = {}
    for name in ["Alpha", "Beta", "Distractor"]:
        canonical = "Canonical" if name == "Alpha" else name
        html = f'<title>{name}</title><link rel="canonical" href="https://en.wikipedia.org/wiki/{canonical}"><p>Evidence for {name}.</p>'
        (directory / f"{name}.html").write_text(html)
        mapping[name] = f"https://en.wikipedia.org/wiki/{name}"
    write_json(tmp_path / "map.json", mapping)
    freeze_snapshot(spec.source_dir, tmp_path / "map.json", "https://snapshot.invalid/frozen", "fixture-v1")
    urls = ["https://en.wikipedia.org/wiki/Alpha", "https://en.wikipedia.org/wiki/Missing" if missing else "https://en.wikipedia.org/wiki/Beta"]
    write_jsonl(spec.questions_path, [{"id": "q", "question": "q", "answer": "g", "normalized_urls": urls}])
    return spec


def test_shared_corpus_redirect_and_mapping(tmp_path):
    spec = setup_snapshot(tmp_path)
    FramesBuilder().build_corpus(spec)
    corpus = [json.loads(line) for line in spec.corpus_path.read_text().splitlines()]
    assert len(corpus) == 3  # Keep distractor even though no question cites it.
    gold = spec.load_questions()[0]["gold_doc_ids"]
    assert article_id("https://en.wikipedia.org/wiki/Canonical") in gold
    assert all(any(d["doc_id"].startswith(g + "_") for d in corpus) for g in gold)
    before = spec.corpus_path.read_bytes()
    FramesBuilder().build_corpus(spec)
    assert spec.corpus_path.read_bytes() == before


def test_missing_coverage_fails_without_dropping_question(tmp_path):
    spec = setup_snapshot(tmp_path, missing=True)
    before = spec.questions_path.read_bytes()
    with pytest.raises(ValueError, match="coverage incomplete"):
        FramesBuilder().build_corpus(spec)
    assert spec.questions_path.read_bytes() == before
    assert not spec.corpus_path.exists()
    assert json.loads((tmp_path / "coverage.json").read_text())["missing"][0]["question_id"] == "q"


def test_changed_snapshot_bytes_rejected(tmp_path):
    spec = setup_snapshot(tmp_path)
    (spec.source_dir / "doc_html" / "Alpha.html").write_text("modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        FramesBuilder().build_corpus(spec)


def test_pilot_final_ids_disjoint_and_reproducible():
    questions = [{"id": f"q{i}"} for i in range(100)]
    a = split_ids(questions, 20261002, 5, 20)
    assert a == split_ids(list(reversed(questions)), 20261002, 5, 20)
    assert not set(a["pilot_ids"]) & set(a["final_ids"])
    assert len(a["unused_ids"]) == 75

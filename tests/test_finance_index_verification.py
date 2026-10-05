import json
from types import SimpleNamespace

import pytest

from scripts import preflight_finance as preflight


class Collection:
    metadata = {"hnsw:space": "cosine"}

    def __init__(self):
        self.texts = {f"doc_p{i}": f"Page {i}" for i in range(1, 4)}
        self.vectors = {key: [1.0] + [0.0] * 1023 for key in self.texts}

    def count(self):
        return len(self.texts)

    def get(self, ids, include):
        found = [key for key in reversed(ids) if key in self.texts]
        return {"ids": found, "documents": [self.texts[key] for key in found],
                "embeddings": [self.vectors[key] for key in found]}


class Embedder:
    def encode_documents(self, texts, batch_size):
        assert batch_size == 2
        return [[1.0] + [0.0] * 1023 for _ in texts]


@pytest.fixture
def index_fixture(tmp_path):
    collection = Collection()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"doc_id": key, "text": text}) for key, text in collection.texts.items()))
    return collection, corpus, Embedder()


def test_verifies_all_texts_and_matches_vectors_by_id(index_fixture):
    collection, corpus, embedder = index_fixture
    result = preflight.verify_existing_index(collection, corpus, embedder)
    assert result["all_ids_and_texts_match"]
    assert result["document_count"] == result["embedding_sample_size"] == 3
    assert all(v["cosine_similarity"] == 1.0 for v in result["embedding_checks"])
    assert result["limitations"]


@pytest.mark.parametrize("damage, message", [
    ("text", "text differs"), ("id", "IDs differ"), ("count", "count differs"),
    ("distance", "cosine distance"), ("vector", "incompatible"),
    ("dimension", "dimension"), ("nan", "Non-finite"),
])
def test_rejects_incompatible_index(index_fixture, damage, message):
    collection, corpus, embedder = index_fixture
    if damage == "text":
        collection.texts["doc_p1"] = "wrong text"
    elif damage == "id":
        collection.texts["wrong_id"] = collection.texts.pop("doc_p1")
        collection.vectors["wrong_id"] = collection.vectors.pop("doc_p1")
    elif damage == "count":
        del collection.texts["doc_p1"]
    elif damage == "distance":
        collection.metadata = {"hnsw:space": "l2"}
    elif damage == "vector":
        collection.vectors["doc_p1"] = [0.0, 1.0] + [0.0] * 1022
    elif damage == "dimension":
        collection.vectors["doc_p1"] = [1.0]
    else:
        collection.vectors["doc_p1"][0] = float("nan")
    with pytest.raises(ValueError, match=message):
        preflight.verify_existing_index(collection, corpus, embedder)


@pytest.mark.parametrize("failure", [None, "content", "retrieval", "data"])
def test_registers_only_after_successful_checks(index_fixture, monkeypatch, failure):
    import os
    import sys

    collection, corpus, embedder = index_fixture
    root = corpus.parent
    (root / "index" / "financebench").mkdir(parents=True)
    (root / "index" / "financebench" / "chroma.sqlite3").touch()
    metadata = {"corpus_count": 3, "corpus.jsonl": {"sha256": preflight.digest(corpus)}}
    monkeypatch.setattr(preflight, "inspect_data", lambda root: (metadata, ["data error"] if failure == "data" else []))
    monkeypatch.setattr(preflight, "version", lambda name: "fixture")
    monkeypatch.setattr(preflight.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HF_HUB_OFFLINE", "original")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    class Retriever:
        def __init__(self, settings):
            assert settings.embedder == "BAAI/bge-m3"
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            self._collection, self._embedder = collection, embedder
            self.document_count = collection.count()

        def search(self, query, top_k):
            return [] if failure == "retrieval" else [SimpleNamespace(doc_id="doc_p1")]

    monkeypatch.setitem(sys.modules, "marlib.retriever", SimpleNamespace(Retriever=Retriever, RetrieverSettings=SimpleNamespace))
    if failure == "content":
        collection.texts["doc_p1"] = "wrong text"
    path = root / "index" / "index_provenance.json"
    report = preflight.check(root, register_existing_index=True)
    assert report["ready"] == (failure is None)
    assert path.exists() == (failure is None)
    assert os.environ["HF_HUB_OFFLINE"] == "original"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    if failure is None:
        saved = path.read_bytes()
        provenance = json.loads(saved)
        assert provenance["provenance_kind"] == "verified_existing_index"
        assert provenance["verification"]["all_ids_and_texts_match"]
        assert provenance["model_revision"] is None
        assert preflight.check(root, register_existing_index=True)["ready"]
        assert path.read_bytes() == saved

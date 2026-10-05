import json
import sys

import pytest

from marlib import cli
from experiments.systems.mas_zero.adapter import MASZeroAdapter


def test_cli_rejects_mas_zero_cl_before_retriever(monkeypatch, capsys):
    monkeypatch.setattr(cli, "discover_adapters", lambda *a: ["mas_zero"])
    monkeypatch.setattr(cli, "get_adapter_class", lambda name: MASZeroAdapter)
    monkeypatch.setattr(sys, "argv", ["bench", "--systems", "mas_zero", "--generation-mode", "one_time"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "supports only" in capsys.readouterr().err


def test_cli_rejects_unsupported_resource_limits(tmp_path, monkeypatch, capsys):
    class Unsupported:
        supports_resource_limits = False
        supported_generation_modes = None
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"fake": {"resource_limits": {"max_requests": 1}}}))
    monkeypatch.setattr(cli, "discover_adapters", lambda *a: ["fake"])
    monkeypatch.setattr(cli, "get_adapter_class", lambda name: Unsupported)
    monkeypatch.setattr(sys, "argv", ["bench", "--systems", "fake", "--adapter-config", str(config)])
    with pytest.raises(SystemExit):
        cli.main()
    assert "not instrumented" in capsys.readouterr().err

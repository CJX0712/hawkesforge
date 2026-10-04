"""CLI 冒烟测试。"""

from __future__ import annotations

import json

from cli import main


def test_cli_models(capsys):
    rc = main(["models"])
    assert rc == 0
    out = capsys.readouterr().out
    for m in ["poisson", "unihawkes", "hawkesfuse"]:
        assert m in out


def test_cli_bench(tmp_path, capsys):
    out = tmp_path / "b.json"
    rc = main(["bench", "--seeds", "1", "--regime", "coupled", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" in data
    assert data["aggregate"]["best"] == "hawkesfuse"

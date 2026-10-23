"""The demo entrypoint runs and writes its outputs."""

import os

from grace import demo, synth


def test_demo_writes_outputs(tmp_path, capsys):
    synth_dir = str(tmp_path / "synthdata")
    out_dir = str(tmp_path / "out")
    synth.write(out_dir=synth_dir, n_cards=200, seed=6)

    exit_code = demo.main(["--out", out_dir, "--synth-dir", synth_dir, "--top", "5"])
    assert exit_code == 0

    assert os.path.getsize(os.path.join(out_dir, "frontier.png")) > 0
    top_path = os.path.join(out_dir, "top_cards.csv")
    assert os.path.getsize(top_path) > 0
    with open(top_path) as handle:
        lines = handle.read().strip().splitlines()
    assert len(lines) <= 6  # header + up to 5 rows

    printed = capsys.readouterr().out
    assert "cards scored:" in printed
    assert "demo finished" in printed

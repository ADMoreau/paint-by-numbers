"""Milestone 6: command line, end to end."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from paint_by_numbers import cli, paints

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "landscape.png"
from core_paints import CORE_NAMES

# Small and quick, and pinned to the core paints so tests do not depend on database size.
# A later --paints in a test overrides this one.
FAST = ["--size", "32", "--steps", "300", "--paints", ",".join(CORE_NAMES)]


def recipe(out: Path) -> dict:
    return json.loads((out / "recipe.json").read_text())


def test_one_command_writes_all_outputs(tmp_path):
    """Spec check: one command on the sample image writes layers, preview,
    recipe and error map. (Small size here to keep the test quick; the README
    shows the default 256 px run.)"""
    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "paint_by_numbers.cli", str(SAMPLE), "--out", str(out), *FAST, "--background", "#7A6A58"],
        capture_output=True, text=True, cwd=ROOT, timeout=600,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    r = recipe(out)
    layers = sorted((out / "layers").glob("*.png"))
    assert len(layers) == len(r["layers"]) > 0
    assert (out / "preview.png").is_file() and (out / "error.png").is_file()
    assert Image.open(out / "preview.png").size == (32, 24)
    assert r["background"]["srgb_hex"] == "#7a6a58"
    assert r["image"]["source"] == str(SAMPLE)
    assert "different paint" in proc.stdout and "bottom first" in proc.stdout and "done in" in proc.stdout


def test_paints_restricts_selection(tmp_path):
    owned = "Titanium White, ultramarine blue, BURNT UMBER, Cadmium Yellow Primrose"
    assert cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, "--paints", owned, "-q"]) == 0
    used = {layer["paint"] for layer in recipe(tmp_path)["layers"]}
    assert used and used <= {"Titanium White", "Ultramarine Blue", "Burnt Umber", "Cadmium Yellow Primrose"}


def test_background_paint_is_layer_zero(tmp_path):
    assert cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, "--background-paint", "burnt umber", "--max-layers", "2", "-q"]) == 0
    r = recipe(tmp_path)
    assert r["background"]["kind"] == "paint"
    assert r["layers"][0] == {**r["layers"][0], "layer": 0, "paint": "Burnt Umber", "role": "ground"}
    assert len(list((tmp_path / "layers").glob("*.png"))) == len(r["layers"]) - 1


def test_background_image(tmp_path):
    ground = tmp_path / "ground.png"
    Image.new("RGB", (20, 20), (90, 80, 70)).save(ground)
    out = tmp_path / "out"
    assert cli.main([str(SAMPLE), "--out", str(out), *FAST, "--background-image", str(ground), "--max-layers", "2", "-q"]) == 0
    assert recipe(out)["background"] == {"kind": "image", "value": str(ground)}


def test_max_layers_and_levels(tmp_path):
    args = [str(SAMPLE), "--out", str(tmp_path), *FAST, "--max-layers", "2", "--t-max", "1", "--levels", "0.5,1", "-q"]
    assert cli.main(args) == 0
    r = recipe(tmp_path)
    assert len(r["layers"]) <= 2
    assert [lv["coats"] for lv in r["thickness_levels"]] == [0.0, 0.5, 1.0]
    assert r["settings"]["solve"]["t_max"] == 1.0


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--paints", "Titanium Whte"], "Did you mean: Titanium White"),
        (["--background-paint", "Mauve"], "no paint named 'Mauve'"),
        (["--background", "#12"], "not a color"),
        (["--levels", "0.5,3"], "--t-max"),
        (["--background", "#fff", "--background-paint", "Carbon Black"], "not allowed with"),
    ],
)
def test_bad_arguments_exit_with_message(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, *extra])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_missing_input(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "nope.png"), "--out", str(tmp_path)])
    assert "input image not found" in capsys.readouterr().err


# ---------------------------------------------------------------- helpers


def test_parse_levels_default_follows_t_max():
    assert cli.parse_levels(None, 2.0) == (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    assert cli.parse_levels(None, 1.0) == (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
    assert cli.parse_levels(None, 3.0)[-1] == 3.0


def test_parse_levels_custom_always_includes_zero():
    assert cli.parse_levels("1, 0.5", 2.0) == (0.0, 0.5, 1.0)


def test_find_paint_ignores_case_and_punctuation():
    db = paints.load_paints()
    assert paints.find_paint(db, "phthalo blue green shade").name == "Phthalo Blue (Green Shade)"
    assert [p.name for p in paints.select_paints(db, "carbon black,Carbon Black, titanium white")] == ["Carbon Black", "Titanium White"]


def test_read_paint_names(tmp_path):
    f = tmp_path / "my_palette.txt"
    f.write_text("# what I own\nTitanium White\nBurnt Umber, Ultramarine Blue  # two on one line\n\n  carbon black \n")
    assert paints.read_paint_names(f) == ["Titanium White", "Burnt Umber", "Ultramarine Blue", "carbon black"]


def test_paints_file_and_max_paints(tmp_path):
    palette_file = tmp_path / "palette.txt"
    palette_file.write_text("Titanium White\nUltramarine Blue\nBurnt Umber\nCadmium Yellow Primrose\nPyrrole Red\n")
    out = tmp_path / "out"
    args = [str(SAMPLE), "--out", str(out), "--size", "32", "--steps", "300",
            "--paints-file", str(palette_file), "--max-paints", "2", "-q"]  # fmt: skip
    assert cli.main(args) == 0
    used = set(recipe(out)["selection"]["paints_used"])
    assert 1 <= len(used) <= 2
    assert used <= {"Titanium White", "Ultramarine Blue", "Burnt Umber", "Cadmium Yellow Primrose", "Pyrrole Red"}


def test_breakdown_flag(tmp_path):
    assert cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, "--max-layers", "2", "--breakdown", "passes", "-q"]) == 0
    assert any((tmp_path / "passes").glob("*.png")) and not (tmp_path / "thickness_overview.png").exists()


def test_missing_paints_file(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(SAMPLE), "--out", str(tmp_path), "--paints-file", str(tmp_path / "nope.txt")])
    assert "paints file not found" in capsys.readouterr().err


def test_max_thicknesses_and_smoothing_flags(tmp_path):
    args = [str(SAMPLE), "--out", str(tmp_path), *FAST, "--max-layers", "2", "--max-thicknesses", "2", "--smoothing", "1.5", "-q"]
    assert cli.main(args) == 0
    r = recipe(tmp_path)
    assert r["settings"]["solve"]["max_thicknesses"] == 2 and r["settings"]["solve"]["refine_smooth"] == 1.5
    assert all(len(layer["thicknesses"]) <= 2 for layer in r["layers"])


def test_max_thicknesses_zero_means_no_limit(tmp_path):
    assert cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, "--max-layers", "1", "--max-thicknesses", "0", "-q"]) == 0
    assert recipe(tmp_path)["settings"]["solve"]["max_thicknesses"] is None


def test_swatch_blocks_format():
    colors = {"full_strength": (10, 20, 30), "on_white": {0.125: (200, 210, 220), 1.0: (100, 110, 120)}}
    text = cli.swatch_blocks(colors)
    assert text.count("\x1b[38;2;") == 3
    assert "\x1b[38;2;10;20;30m" in text and "\x1b[38;2;100;110;120m" in text


def test_no_color_codes_when_not_a_terminal(tmp_path, capsys):
    assert cli.main([str(SAMPLE), "--out", str(tmp_path), *FAST, "--max-layers", "1", "-q"]) == 0
    assert "\x1b[" not in capsys.readouterr().out

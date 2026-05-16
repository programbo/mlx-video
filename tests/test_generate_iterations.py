"""Tests for native iterative generation options."""

from pathlib import Path


def test_wan_iteration_output_path_includes_seed_steps_shift_and_size(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        prefix="demo-",
        suffix="test",
        mode="t2v",
        seed=123,
        steps=8,
        shift=5.0,
        width=672,
        height=1024,
    )

    assert path == tmp_path / "demo-wan-t2v-seed123-s8-sh5-672x1024-test.mp4"


def test_ltx_iteration_output_path_includes_seed_steps_and_size(tmp_path):
    from mlx_video.models.ltx_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        prefix="demo-",
        suffix="test",
        seed=123,
        steps=30,
        width=512,
        height=768,
    )

    assert path == tmp_path / "demo-ltx-seed123-s30-512x768-test.mp4"


def test_wan_main_runs_incrementing_iterations(monkeypatch, tmp_path):
    from mlx_video.models.wan_2 import generate

    calls = []

    def fake_generate_video(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(generate, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate",
            "--model-dir",
            "model",
            "--prompt",
            "prompt",
            "--seed",
            "10",
            "--steps",
            "8",
            "--shift",
            "5",
            "--width",
            "672",
            "--height",
            "1024",
            "--iterations",
            "2",
            "--output-path",
            str(tmp_path),
            "--output-prefix",
            "p-",
            "--output-suffix",
            "s",
        ],
    )

    generate.main()

    assert [call["seed"] for call in calls] == [10, 11]
    assert [Path(call["output_path"]).name for call in calls] == [
        "p-wan-t2v-seed10-s8-sh5-672x1024-s.mp4",
        "p-wan-t2v-seed11-s8-sh5-672x1024-s.mp4",
    ]


def test_ltx_main_runs_incrementing_iterations(monkeypatch, tmp_path):
    from mlx_video.models.ltx_2 import generate

    calls = []

    def fake_generate_video(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(generate, "generate_video", fake_generate_video)
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate",
            "--prompt",
            "prompt",
            "--seed",
            "20",
            "--steps",
            "30",
            "--width",
            "512",
            "--height",
            "768",
            "--iterations",
            "2",
            "--output-path",
            str(tmp_path),
            "--output-prefix",
            "p-",
            "--output-suffix",
            "s",
        ],
    )

    generate.main()

    assert [call["seed"] for call in calls] == [20, 21]
    assert [Path(call["output_path"]).name for call in calls] == [
        "p-ltx-seed20-s30-512x768-s.mp4",
        "p-ltx-seed21-s30-512x768-s.mp4",
    ]

"""Tests for native iterative generation options."""

from pathlib import Path

import pytest


def test_wan_iteration_output_path_includes_seed_steps_shift_and_size(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        template="demo-wan-{mode}-seed{seed}-s{steps}-sh{shift}-{size}-test.mp4",
        mode="t2v",
        seed=123,
        steps=8,
        shift=5.0,
        width=672,
        height=1024,
        frames=81,
        fps=16,
        iteration=0,
    )

    assert path == tmp_path / "demo-wan-t2v-seed123-s8-sh5-672x1024-test.mp4"


def test_ltx_iteration_output_path_includes_seed_steps_and_size(tmp_path):
    from mlx_video.models.ltx_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        template="demo-ltx-seed{seed}-s{steps}-{size}-test.mp4",
        seed=123,
        steps=30,
        width=512,
        height=768,
        frames=100,
        fps=24,
        iteration=0,
    )

    assert path == tmp_path / "demo-ltx-seed123-s30-512x768-test.mp4"


def test_wan_iteration_output_path_supports_nested_template_and_format_specs(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        template="{mode}/run-{iteration1:03d}/seed{seed}-{steps_part}-{shift_part}-{size}",
        mode="i2v",
        seed=123,
        steps=8,
        shift=5.0,
        width=672,
        height=1024,
        frames=81,
        fps=16,
        iteration=4,
    )

    assert path == tmp_path / "i2v/run-005/seed123-s8-sh5-672x1024.mp4"
    assert path.parent.is_dir()


def test_ltx_iteration_output_path_supports_custom_template(tmp_path):
    from mlx_video.models.ltx_2.generate import _iteration_output_path

    path = _iteration_output_path(
        tmp_path,
        template="{model}/iter{iteration1:02d}/seed{seed}-{fps}fps.mp4",
        seed=20,
        steps=30,
        width=512,
        height=768,
        frames=100,
        fps=24,
        iteration=1,
    )

    assert path == tmp_path / "ltx/iter02/seed20-24fps.mp4"


def test_output_template_rejects_absolute_paths(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path
    from mlx_video.utils import OutputTemplateError

    with pytest.raises(OutputTemplateError, match="relative path"):
        _iteration_output_path(
            tmp_path,
            template="/tmp/{seed}.mp4",
            mode="t2v",
            seed=123,
            steps=8,
            shift=5.0,
            width=672,
            height=1024,
            frames=81,
            fps=16,
            iteration=0,
        )


def test_output_template_rejects_path_traversal(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path
    from mlx_video.utils import OutputTemplateError

    with pytest.raises(OutputTemplateError, match="\\.\\."):
        _iteration_output_path(
            tmp_path,
            template="../{seed}.mp4",
            mode="t2v",
            seed=123,
            steps=8,
            shift=5.0,
            width=672,
            height=1024,
            frames=81,
            fps=16,
            iteration=0,
        )


def test_output_template_rejects_unknown_fields(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path
    from mlx_video.utils import OutputTemplateError

    with pytest.raises(OutputTemplateError, match="prompt_slug"):
        _iteration_output_path(
            tmp_path,
            template="{prompt_slug}.mp4",
            mode="t2v",
            seed=123,
            steps=8,
            shift=5.0,
            width=672,
            height=1024,
            frames=81,
            fps=16,
            iteration=0,
        )


def test_output_template_rejects_malformed_templates(tmp_path):
    from mlx_video.models.wan_2.generate import _iteration_output_path
    from mlx_video.utils import OutputTemplateError

    with pytest.raises(OutputTemplateError, match="invalid output template"):
        _iteration_output_path(
            tmp_path,
            template="{seed",
            mode="t2v",
            seed=123,
            steps=8,
            shift=5.0,
            width=672,
            height=1024,
            frames=81,
            fps=16,
            iteration=0,
        )


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
        ],
    )

    generate.main()

    assert [call["seed"] for call in calls] == [10, 11]
    assert [Path(call["output_path"]).name for call in calls] == [
        "wan-t2v-seed10-s8-sh5-672x1024.mp4",
        "wan-t2v-seed11-s8-sh5-672x1024.mp4",
    ]


def test_wan_main_uses_output_template_for_single_run(monkeypatch, tmp_path):
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
            "--output-path",
            str(tmp_path),
            "--output-template",
            "{mode}/iter{iteration1:03d}/seed{seed}-{size}",
        ],
    )

    generate.main()

    assert Path(calls[0]["output_path"]) == (
        tmp_path / "t2v/iter001/seed10-672x1024.mp4"
    )


def test_wan_main_uses_default_template_for_single_run_output_directory(
    monkeypatch, tmp_path
):
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
            "--output-path",
            str(tmp_path),
        ],
    )

    generate.main()

    assert Path(calls[0]["output_path"]) == (
        tmp_path / "wan-t2v-seed10-s8-sh5-672x1024.mp4"
    )


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
        ],
    )

    generate.main()

    assert [call["seed"] for call in calls] == [20, 21]
    assert [Path(call["output_path"]).name for call in calls] == [
        "ltx-seed20-s30-512x768.mp4",
        "ltx-seed21-s30-512x768.mp4",
    ]


def test_ltx_main_uses_default_template_for_single_run_output_directory(
    monkeypatch, tmp_path
):
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
            "--output-path",
            str(tmp_path),
        ],
    )

    generate.main()

    assert Path(calls[0]["output_path"]) == tmp_path / "ltx-seed20-s30-512x768.mp4"

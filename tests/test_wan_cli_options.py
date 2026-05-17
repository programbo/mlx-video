"""Tests for Wan generation CLI option plumbing."""

import argparse
import json

import numpy as np
import pytest


def test_wan_parser_accepts_output_fps():
    from mlx_video.models.wan_2.generate import build_parser

    args = build_parser().parse_args(
        [
            "--model-dir",
            "model",
            "--prompt",
            "prompt",
            "--fps",
            "24",
        ]
    )

    assert args.fps == 24


def test_wan_parser_accepts_repeatable_config():
    from mlx_video.models.wan_2.generate import build_parser

    args = build_parser().parse_args(
        [
            "--config",
            "first.yaml",
            "--config",
            "second.json",
            "--config",
            "third.yml",
        ]
    )

    assert args.config == ["first.yaml", "second.json", "third.yml"]


def test_wan_parser_allows_model_dir_and_prompt_from_config():
    from mlx_video.models.wan_2.generate import build_parser

    args = build_parser().parse_args(["--config", "run.yaml"])

    assert args.config == ["run.yaml"]
    assert args.model_dir is None
    assert args.prompt is None


def test_load_generation_config_supports_json(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "model_dir": "model",
                "prompt": "prompt",
                "width": 832,
                "lora": [["lora.safetensors", 0.8]],
            }
        )
    )

    assert _load_generation_config(path) == {
        "model_dir": "model",
        "prompt": "prompt",
        "width": 832,
        "lora": [["lora.safetensors", 0.8]],
    }


def test_load_generation_config_supports_yaml(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.yaml"
    path.write_text(
        "\n".join(
            [
                "model_dir: model",
                "prompt: prompt",
                "height: 704",
                "guide_scale: '3.0,4.0'",
            ]
        )
    )

    assert _load_generation_config(path) == {
        "model_dir": "model",
        "prompt": "prompt",
        "height": 704,
        "guide_scale": "3.0,4.0",
    }


def test_load_generation_config_rejects_unknown_key(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.json"
    path.write_text(json.dumps({"model_dir": "model", "prompt": "prompt", "bad": True}))

    with pytest.raises(SystemExit, match="unknown config key"):
        _load_generation_config(path)


def test_load_generation_config_rejects_non_string_unknown_yaml_key(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.yaml"
    path.write_text("1: bad\nmodel_dir: model\nprompt: prompt\n")

    with pytest.raises(SystemExit, match="unknown config key"):
        _load_generation_config(path)


def test_load_generation_config_rejects_unsupported_extension(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.toml"
    path.write_text("model_dir = 'model'")

    with pytest.raises(SystemExit, match="Unsupported config extension"):
        _load_generation_config(path)


def test_load_generation_config_rejects_missing_file(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "missing.yaml"

    with pytest.raises(SystemExit, match="could not read config"):
        _load_generation_config(path)


def test_load_generation_config_rejects_non_object_yaml(tmp_path):
    from mlx_video.models.wan_2.generate import _load_generation_config

    path = tmp_path / "run.yaml"
    path.write_text("- prompt\n- model\n")

    with pytest.raises(SystemExit, match="config must be a JSON/YAML object"):
        _load_generation_config(path)


def test_config_resolution_applies_cli_overrides(tmp_path):
    from mlx_video.models.wan_2.generate import (
        _explicit_cli_dests,
        _resolve_generation_runs,
        build_parser,
    )

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "model_dir": "from-config",
                "prompt": "config prompt",
                "width": 832,
                "steps": 8,
                "output_path": "config.mp4",
            }
        )
    )
    argv = [
        "--config",
        str(path),
        "--prompt",
        "cli prompt",
        "--width",
        "640",
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = _resolve_generation_runs(parser, args, _explicit_cli_dests(parser, argv))

    assert len(runs) == 1
    assert runs[0].model_dir == "from-config"
    assert runs[0].prompt == "cli prompt"
    assert runs[0].width == 640
    assert runs[0].steps == 8
    assert runs[0].output_path == "config.mp4"


def test_config_resolution_accepts_refiner_start(tmp_path):
    from mlx_video.models.wan_2.generate import _resolve_generation_runs, build_parser

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "model_dir": "model",
                "prompt": "prompt",
                "steps": 8,
                "refiner_start": 0.125,
            }
        )
    )
    parser = build_parser()
    args = parser.parse_args(["--config", str(path)])
    runs = _resolve_generation_runs(parser, args, set())

    assert runs[0].refiner_start == 0.125


def test_config_resolution_accepts_sigma_options(tmp_path):
    from mlx_video.models.wan_2.generate import _resolve_generation_runs, build_parser

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "model_dir": "model",
                "prompt": "prompt",
                "sigma_schedule": "comfy-simple",
            }
        )
    )
    parser = build_parser()
    args = parser.parse_args(["--config", str(path)])
    runs = _resolve_generation_runs(parser, args, set())

    assert runs[0].sigma_schedule == "comfy-simple"


def test_config_resolution_accepts_reference_bridge_options(tmp_path):
    from mlx_video.models.wan_2.generate import _resolve_generation_runs, build_parser

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "model_dir": "model",
                "prompt": "prompt",
                "t2v_lightning_preset": True,
                "positive_conditioning_npz": "positive.npz",
                "negative_conditioning_npz": "negative.npz",
                "dump_text_conditioning_npz": "text.npz",
                "dump_final_latents_npz": "final.npz",
                "initial_latents_npz": "initial.npz",
            }
        )
    )
    parser = build_parser()
    args = parser.parse_args(["--config", str(path)])
    runs = _resolve_generation_runs(parser, args, set())

    assert runs[0].t2v_lightning_preset is True
    assert runs[0].positive_conditioning_npz == "positive.npz"
    assert runs[0].negative_conditioning_npz == "negative.npz"
    assert runs[0].dump_text_conditioning_npz == "text.npz"
    assert runs[0].dump_final_latents_npz == "final.npz"
    assert runs[0].initial_latents_npz == "initial.npz"


def test_config_resolution_detects_equals_style_cli_overrides(tmp_path):
    from mlx_video.models.wan_2.generate import (
        _explicit_cli_dests,
        _resolve_generation_runs,
        build_parser,
    )

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({"model_dir": "model", "prompt": "config prompt", "width": 832})
    )
    argv = ["--config", str(path), "--width=640"]
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = _resolve_generation_runs(parser, args, _explicit_cli_dests(parser, argv))

    assert runs[0].width == 640


def test_config_resolution_plans_multiple_runs_in_order(tmp_path):
    from mlx_video.models.wan_2.generate import (
        _explicit_cli_dests,
        _resolve_generation_runs,
        build_parser,
    )

    first = tmp_path / "first.yaml"
    second = tmp_path / "second.json"
    first.write_text("model_dir: model-a\nprompt: first\niterations: 2\n")
    second.write_text(json.dumps({"model_dir": "model-b", "prompt": "second"}))

    argv = ["--config", str(first), "--config", str(second), "--height", "512"]
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = _resolve_generation_runs(parser, args, _explicit_cli_dests(parser, argv))

    assert [run.prompt for run in runs] == ["first", "second"]
    assert [run.model_dir for run in runs] == ["model-a", "model-b"]
    assert [run.height for run in runs] == [512, 512]
    assert runs[0].iterations == 2
    assert runs[1].iterations == 1


def test_config_resolution_rejects_missing_model_dir(tmp_path):
    from mlx_video.models.wan_2.generate import _resolve_generation_runs, build_parser

    path = tmp_path / "run.json"
    path.write_text(json.dumps({"prompt": "prompt"}))
    parser = build_parser()
    args = parser.parse_args(["--config", str(path)])

    with pytest.raises(SystemExit, match="--model-dir is required"):
        _resolve_generation_runs(parser, args, set())


def test_config_resolution_rejects_missing_prompt(tmp_path):
    from mlx_video.models.wan_2.generate import _resolve_generation_runs, build_parser

    path = tmp_path / "run.json"
    path.write_text(json.dumps({"model_dir": "model"}))
    parser = build_parser()
    args = parser.parse_args(["--config", str(path)])

    with pytest.raises(SystemExit, match="--prompt is required"):
        _resolve_generation_runs(parser, args, set())


def test_parse_lora_args_rejects_invalid_config_shape():
    from mlx_video.models.wan_2.generate import _parse_lora_args

    with pytest.raises(SystemExit, match="lora entries must be"):
        _parse_lora_args(["not-a-pair"], "lora")


def test_parse_lora_args_rejects_invalid_strength():
    from mlx_video.models.wan_2.generate import _parse_lora_args

    with pytest.raises(SystemExit, match="lora strength must be a number"):
        _parse_lora_args([["path.safetensors", "strong"]], "lora")


def test_parse_guide_scale_accepts_config_list():
    from mlx_video.models.wan_2.generate import _parse_guide_scale

    assert _parse_guide_scale([3.0, 4.0]) == (3.0, 4.0)


def test_parse_guide_scale_rejects_empty_config_list():
    from mlx_video.models.wan_2.generate import _parse_guide_scale

    with pytest.raises(SystemExit, match="guide_scale must not be empty"):
        _parse_guide_scale([])


def test_wan_parser_accepts_refiner_start():
    from mlx_video.models.wan_2.generate import build_parser

    args = build_parser().parse_args(
        [
            "--model-dir",
            "model",
            "--prompt",
            "prompt",
            "--refiner-start",
            "0.125",
        ]
    )

    assert args.refiner_start == 0.125


def test_resolve_refiner_start_accepts_fractional_boundary():
    from mlx_video.models.wan_2.generate import _resolve_refiner_start

    assert _resolve_refiner_start(0.125, 8) == (2, 1)
    assert _resolve_refiner_start(0.2, 8) == (3, 2)


def test_resolve_refiner_start_accepts_one_based_steps():
    from mlx_video.models.wan_2.generate import _resolve_refiner_start

    assert _resolve_refiner_start(1, 8) == (1, 0)
    assert _resolve_refiner_start(2, 8) == (2, 1)
    assert _resolve_refiner_start(9, 8) == (9, 8)


def test_resolve_refiner_start_rejects_ambiguous_or_out_of_range_values():
    from mlx_video.models.wan_2.generate import _resolve_refiner_start

    for value in (0, 1.5, 10, -0.25):
        with pytest.raises(ValueError, match="--refiner-start"):
            _resolve_refiner_start(value, 8)


def test_wan_parser_accepts_scheduler_choices():
    from mlx_video.models.wan_2.generate import build_parser

    parser = build_parser()

    for scheduler in ("euler", "dpm++", "unipc"):
        args = parser.parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--scheduler",
                scheduler,
            ]
        )
        assert args.scheduler == scheduler


def test_wan_parser_rejects_unknown_scheduler():
    from mlx_video.models.wan_2.generate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--scheduler",
                "unknown",
            ]
        )


def test_wan_parser_accepts_sigma_schedule_choices():
    from mlx_video.models.wan_2.generate import build_parser

    parser = build_parser()

    for sigma_schedule in ("official", "comfy-simple"):
        args = parser.parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--sigma-schedule",
                sigma_schedule,
            ]
        )
        assert args.sigma_schedule == sigma_schedule


def test_wan_parser_rejects_unknown_sigma_schedule():
    from mlx_video.models.wan_2.generate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--sigma-schedule",
                "unknown",
            ]
        )


def test_wan_parser_accepts_noise_source_options():
    from mlx_video.models.wan_2.generate import build_parser

    args = build_parser().parse_args(
        [
            "--model-dir",
            "model",
            "--prompt",
            "prompt",
            "--noise-source",
            "torch",
            "--torch-python",
            "/usr/bin/python3",
        ]
    )

    assert args.noise_source == "torch"
    assert args.torch_python == "/usr/bin/python3"


def test_t2v_lightning_preset_sets_t2v_sampling_defaults():
    from mlx_video.models.wan_2.generate import (
        REFERENCE_NEGATIVE_PROMPT,
        WAN22_T2V_LIGHTNING_HIGH_PATH,
        WAN22_T2V_LIGHTNING_LOW_PATH,
        _resolve_generation_runs,
        build_parser,
    )

    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-dir",
            "model",
            "--prompt",
            "prompt",
            "--t2v-lightning-preset",
        ]
    )
    runs = _resolve_generation_runs(
        parser, args, {"model_dir", "prompt", "t2v_lightning_preset"}
    )

    assert runs[0].steps == 8
    assert runs[0].guide_scale == 1
    assert runs[0].shift == 5
    assert runs[0].scheduler == "euler"
    assert runs[0].sigma_schedule == "comfy-simple"
    assert runs[0].fps == 24
    assert runs[0].refiner_start == 0.125
    assert runs[0].noise_source == "torch"
    assert runs[0].negative_prompt == REFERENCE_NEGATIVE_PROMPT
    assert runs[0].lora_high == [(WAN22_T2V_LIGHTNING_HIGH_PATH, 1.0)]
    assert runs[0].lora_low == [(WAN22_T2V_LIGHTNING_LOW_PATH, 1.0)]


def test_t2v_lightning_preset_preserves_explicit_overrides():
    from mlx_video.models.wan_2.generate import (
        _explicit_cli_dests,
        _resolve_generation_runs,
        build_parser,
    )

    argv = [
        "--model-dir",
        "model",
        "--prompt",
        "prompt",
        "--t2v-lightning-preset",
        "--scheduler",
        "unipc",
        "--noise-source",
        "mlx",
        "--negative-prompt",
        "custom negative",
        "--lora-high",
        "custom-high.safetensors",
        "0.5",
    ]
    parser = build_parser()
    args = parser.parse_args(argv)
    runs = _resolve_generation_runs(parser, args, _explicit_cli_dests(parser, argv))

    assert runs[0].scheduler == "unipc"
    assert runs[0].noise_source == "mlx"
    assert runs[0].negative_prompt == "custom negative"
    assert runs[0].lora_high == [["custom-high.safetensors", "0.5"]]


def test_wan_parser_rejects_unknown_noise_source():
    from mlx_video.models.wan_2.generate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--noise-source",
                "unknown",
            ]
        )


def test_torch_randn_is_seed_deterministic():
    pytest.importorskip("torch")

    from mlx_video.models.wan_2.generate import _torch_randn

    first = np.array(_torch_randn((2, 3), 123))
    second = np.array(_torch_randn((2, 3), 123))
    different = np.array(_torch_randn((2, 3), 124))

    np.testing.assert_allclose(first, second)
    assert first.shape == (2, 3)
    assert not np.allclose(first, different)


def test_torch_randn_matches_one_batch_torch_seed_semantics():
    torch = pytest.importorskip("torch")

    from mlx_video.models.wan_2.generate import _torch_randn

    shape = (2, 3, 4)
    generator = torch.manual_seed(123)
    expected = torch.randn(
        (1, *shape), dtype=torch.float32, device="cpu", generator=generator
    ).numpy()[0]

    np.testing.assert_allclose(np.array(_torch_randn(shape, 123)), expected)


def test_npz_bridge_loaders_accept_supported_keys_and_strip_batch(tmp_path):
    from mlx_video.models.wan_2.generate import (
        _load_latents_npz,
        _load_text_conditioning_npz,
    )

    conditioning_path = tmp_path / "conditioning.npz"
    latents_path = tmp_path / "latents.npz"
    np.savez(conditioning_path, conditioning=np.ones((1, 512, 4096), dtype=np.float32))
    np.savez(latents_path, samples=np.ones((1, 48, 1, 40, 22), dtype=np.float32))

    assert tuple(_load_text_conditioning_npz(conditioning_path).shape) == (512, 4096)
    assert tuple(_load_latents_npz(latents_path).shape) == (48, 1, 40, 22)


def test_npz_bridge_loaders_reject_bad_shapes(tmp_path):
    from mlx_video.models.wan_2.generate import (
        _load_latents_npz,
        _load_text_conditioning_npz,
    )

    conditioning_path = tmp_path / "conditioning.npz"
    latents_path = tmp_path / "latents.npz"
    np.savez(conditioning_path, positive=np.ones((2, 512, 4096), dtype=np.float32))
    np.savez(latents_path, latents=np.ones((48, 40, 22), dtype=np.float32))

    with pytest.raises(ValueError, match="must have shape"):
        _load_text_conditioning_npz(conditioning_path)
    with pytest.raises(ValueError, match="must have shape"):
        _load_latents_npz(latents_path)


def test_npz_bridge_dumpers_write_stable_keys(tmp_path):
    import mlx.core as mx

    from mlx_video.models.wan_2.generate import (
        _dump_latents_npz,
        _dump_text_conditioning_npz,
    )

    class Config:
        vae_z_dim = 16

    text_path = tmp_path / "text.npz"
    latents_path = tmp_path / "final.npz"
    _dump_text_conditioning_npz(
        text_path,
        mx.array(np.ones((2, 3), dtype=np.float32)),
        mx.array(np.zeros((2, 3), dtype=np.float32)),
    )
    _dump_latents_npz(
        latents_path,
        mx.array(np.ones((16, 1, 2, 3), dtype=np.float32)),
        Config(),
    )

    text = np.load(text_path)
    latents = np.load(latents_path)
    assert sorted(text.files) == ["negative", "positive"]
    assert sorted(latents.files) == ["latents_model", "latents_vae_bcthw"]


def test_wan_parser_accepts_tiling_modes():
    from mlx_video.models.wan_2.generate import build_parser

    parser = build_parser()

    for tiling in (
        "auto",
        "none",
        "default",
        "aggressive",
        "conservative",
        "spatial",
        "temporal",
    ):
        args = parser.parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--tiling",
                tiling,
            ]
        )
        assert args.tiling == tiling


def test_wan_parser_accepts_legacy_vae_decode_flag():
    from mlx_video.models.wan_2.generate import build_parser

    base = ["--model-dir", "model", "--prompt", "prompt"]

    default = build_parser().parse_args(base)
    legacy = build_parser().parse_args(base + ["--legacy-vae-decode"])

    assert default.legacy_vae_decode is False
    assert legacy.legacy_vae_decode is True


def test_wan_parser_rejects_unknown_tiling_mode():
    from mlx_video.models.wan_2.generate import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-dir",
                "model",
                "--prompt",
                "prompt",
                "--tiling",
                "unknown",
            ]
        )


def test_wan_parser_accepts_last_frame_flags():
    from mlx_video.models.wan_2.generate import build_parser

    base = ["--model-dir", "model", "--prompt", "prompt"]

    default = build_parser().parse_args(base)
    enabled = build_parser().parse_args(base + ["--output-last-frame"])
    disabled = build_parser().parse_args(base + ["--no-output-last-frame"])

    assert default.output_last_frame is None
    assert enabled.output_last_frame is True
    assert disabled.output_last_frame is False


def test_last_frame_default_only_for_single_frame_runs():
    from mlx_video.utils import should_output_last_frame

    assert should_output_last_frame(None, 1) is True
    assert should_output_last_frame(None, 5) is False
    assert should_output_last_frame(True, 5) is True
    assert should_output_last_frame(False, 1) is False


def test_crop_decoded_video_keeps_latest_requested_frames():
    from mlx_video.models.wan_2.generate import _crop_decoded_video

    video = np.arange(4, dtype=np.uint8).reshape(4, 1, 1, 1)

    assert _crop_decoded_video(video, 1).shape[0] == 1
    assert int(_crop_decoded_video(video, 1)[0, 0, 0, 0]) == 3
    assert _crop_decoded_video(video, 2)[:, 0, 0, 0].tolist() == [2, 3]


def test_save_last_frame_png(tmp_path):
    from PIL import Image

    from mlx_video.utils import save_last_frame_png

    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    frames[-1, :, :, 0] = 255
    output_path = tmp_path / "sample.mp4"

    png_path = save_last_frame_png(frames, output_path)

    assert png_path == tmp_path / "sample.png"
    saved = np.array(Image.open(png_path))
    assert int(saved[:, :, 0].max()) == 255


def test_wan_save_video_uses_requested_fps(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    from mlx_video.models.wan_2.postprocess import save_video

    output_path = tmp_path / "fps-test.mp4"
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    save_video(frames, str(output_path), fps=24)

    assert output_path.exists()

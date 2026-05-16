"""Tests for Wan generation CLI option plumbing."""

import argparse

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

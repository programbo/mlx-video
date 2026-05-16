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


def test_wan_save_video_uses_requested_fps(tmp_path):
    pytest.importorskip("imageio")
    pytest.importorskip("imageio_ffmpeg")

    from mlx_video.models.wan_2.postprocess import save_video

    output_path = tmp_path / "fps-test.mp4"
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)

    save_video(frames, str(output_path), fps=24)

    assert output_path.exists()

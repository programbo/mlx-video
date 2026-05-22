"""Wan2.2 Text-to-Video generation pipeline for MLX."""

import argparse
import copy
import gc
import json
import math
import random
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from tqdm import tqdm

from mlx_video.models.wan_2.i2v_utils import build_i2v_mask, preprocess_image
from mlx_video.models.wan_2.utils import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)
from mlx_video.models.wan_2.postprocess import save_video
from mlx_video.utils import (
    OutputTemplateError,
    format_output_value,
    render_output_template,
    save_last_frame_png,
    should_output_last_frame,
)


class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


# Backward-compat alias (tests and external code may use the old name)
_build_i2v_mask = build_i2v_mask

REFERENCE_NEGATIVE_PROMPT = (
    "overly vivid colors, overexposure, static, blurry details, subtitles, "
    "watermark, artwork, painting, still image, overall grey, worst quality, "
    "low quality, JPEG compression artifacts, ugly, deformed, extra fingers, "
    "poorly drawn hands, poorly drawn faces, distorted, disfigured, malformed "
    "limbs, fused fingers, motionless, cluttered background, three legs, "
    "crowded background, walking backwards"
)
WAN22_T2V_LIGHTNING_HIGH_PATH = (
    "/Volumes/Jove/Models/loras/WAN2.2/T2V/Lightning 4Steps A14B FP16 HIGH.safetensors"
)
WAN22_T2V_LIGHTNING_LOW_PATH = (
    "/Volumes/Jove/Models/loras/WAN2.2/T2V/Lightning 4Steps A14B FP16 LOW.safetensors"
)
CONFIG_KEYS = {
    "model_dir",
    "prompt",
    "image",
    "negative_prompt",
    "no_negative_prompt",
    "width",
    "height",
    "num_frames",
    "steps",
    "guide_scale",
    "shift",
    "refiner_start",
    "seed",
    "output_path",
    "fps",
    "output_last_frame",
    "scheduler",
    "sigma_schedule",
    "noise_source",
    "torch_python",
    "t2v_lightning_preset",
    "positive_conditioning_npz",
    "negative_conditioning_npz",
    "dump_text_conditioning_npz",
    "dump_final_latents_npz",
    "initial_latents_npz",
    "lora",
    "lora_high",
    "lora_low",
    "no_lora",
    "no_lora_high",
    "no_lora_low",
    "tiling",
    "legacy_vae_decode",
    "no_compile",
    "trim_first_frames",
    "debug_latents",
    "iterations",
    "iteration_seed",
    "output_template",
}

REPO_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"

CHOICES = {
    "scheduler": {"euler", "dpm++", "unipc"},
    "sigma_schedule": {"official", "comfy-simple"},
    "noise_source": {"mlx", "torch"},
    "tiling": {
        "auto",
        "none",
        "default",
        "aggressive",
        "conservative",
        "spatial",
        "temporal",
    },
    "iteration_seed": {"same", "increment", "random"},
}


def _crop_decoded_video(video: np.ndarray, num_frames: int) -> np.ndarray:
    """Return exactly the requested frame count, keeping the latest frames."""
    if video.shape[0] <= num_frames:
        return video
    return video[-num_frames:]


def _iteration_seed(base_seed: int, iteration: int, strategy: str) -> int:
    """Resolve the concrete seed for an iteration."""
    if strategy == "same":
        return base_seed
    if strategy == "increment":
        if base_seed < 0:
            base_seed = random.randint(0, 2**32 - 1)
        return base_seed + iteration
    if strategy == "random":
        return random.randint(0, 2**32 - 1)
    raise ValueError(f"Unsupported iteration seed strategy: {strategy}")


def _iteration_output_path(
    output_dir: str | Path,
    *,
    template: str,
    mode: str,
    seed: int,
    steps: int | None,
    shift: float | None,
    width: int,
    height: int,
    frames: int,
    fps: int,
    iteration: int,
    prompt_index: int = 0,
    prompt_count: int = 1,
) -> Path:
    """Build a Wan output path from a filename template."""
    fields = {
        "model": "wan",
        "mode": mode,
        "seed": seed,
        "steps": format_output_value(steps),
        "shift": format_output_value(shift),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "iteration": iteration,
        "iteration1": iteration + 1,
        "prompt_index": prompt_index,
        "prompt_index1": prompt_index + 1,
        "prompt_count": prompt_count,
        "size": f"{width}x{height}",
        "mode_part": f"{mode}-" if mode else "",
        "steps_part": f"s{format_output_value(steps)}",
        "shift_part": f"sh{format_output_value(shift)}" if shift is not None else "",
    }
    return render_output_template(output_dir, template, fields)


def _parser_defaults(parser: argparse.ArgumentParser) -> dict:
    defaults = vars(parser.parse_args([]))
    defaults.pop("config", None)
    return defaults


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit_dests = set()
    for arg in argv:
        option = arg.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest not in {None, "config"}:
            explicit_dests.add(dest)
    return explicit_dests


def _load_generation_config(path: str | Path) -> dict:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".json":
            with open(path) as f:
                data = json.load(f)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            with open(path) as f:
                data = yaml.safe_load(f)
        else:
            raise SystemExit(
                f"Unsupported config extension for {path}: expected .json, .yaml, or .yml"
            )
    except OSError as exc:
        raise SystemExit(f"{path}: could not read config: {exc}") from exc

    if not isinstance(data, dict):
        raise SystemExit(f"{path}: config must be a JSON/YAML object")

    unknown = sorted(str(key) for key in data if key not in CONFIG_KEYS)
    if unknown:
        keys = ", ".join(unknown)
        raise SystemExit(f"{path}: unknown config key(s): {keys}")

    return data


def _resolve_generation_config_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path

    path_text = str(path)
    if raw_path.parent != Path(".") or path_text.startswith(("./", "../")):
        return raw_path

    return REPO_CONFIG_DIR / raw_path


def _load_generation_config_with_defaults(path: str | Path) -> tuple[dict, set[str]]:
    path = _resolve_generation_config_path(path)
    config = _load_generation_config(path)
    user_keys = set(config.keys())

    default_path = path.with_name("_default.yaml")
    if path.name != "_default.yaml" and default_path.exists():
        defaults = _load_generation_config(default_path)
        return defaults | config, user_keys

    return config, user_keys


def _parse_guide_scale(value):
    if value is None or isinstance(value, (int, float, tuple)):
        return value
    if isinstance(value, list):
        if not value:
            raise SystemExit("guide_scale must not be empty")
        return tuple(float(x) for x in value) if len(value) > 1 else float(value[0])
    parts = [float(x) for x in str(value).split(",")]
    return tuple(parts) if len(parts) > 1 else parts[0]


def _parse_lora_args(lora_list, name: str):
    if not lora_list:
        return None

    parsed = []
    for item in lora_list:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SystemExit(f"{name} entries must be [path, strength] pairs")
        path, strength = item
        try:
            parsed.append((path, float(strength)))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"{name} strength must be a number") from exc
    return parsed


def apply_t2v_lightning_defaults(
    values: dict, user_keys: set[str] | None = None
) -> None:
    """Apply Wan2.2 T2V Lightning reference defaults without overriding user values."""
    if not values.get("t2v_lightning_preset"):
        return
    if values.get("image"):
        raise SystemExit("--t2v-lightning-preset is only supported for T2V runs")

    user_keys = user_keys or set()
    defaults = {
        "steps": 8,
        "guide_scale": 1,
        "shift": 5,
        "scheduler": "euler",
        "sigma_schedule": "comfy-simple",
        "fps": 24,
        "refiner_start": 0.125,
        "noise_source": "torch",
        "negative_prompt": REFERENCE_NEGATIVE_PROMPT,
        "lora_high": [(WAN22_T2V_LIGHTNING_HIGH_PATH, 1.0)],
        "lora_low": [(WAN22_T2V_LIGHTNING_LOW_PATH, 1.0)],
    }
    for key, default in defaults.items():
        if key not in user_keys:
            values[key] = default


def _resolve_refiner_start(refiner_start: float | int, steps: int) -> tuple[int, int]:
    """Resolve refiner_start to one-based LNE start step and HNE step count."""
    value = float(refiner_start)
    if 0 < value < 1:
        high_noise_steps = math.ceil(steps * value)
        return high_noise_steps + 1, high_noise_steps
    if value.is_integer() and 1 <= value <= steps + 1:
        lne_start_step = int(value)
        return lne_start_step, lne_start_step - 1
    raise ValueError(
        "--refiner-start must be a fraction in (0, 1) or an integer step in "
        f"[1, {steps + 1}]"
    )


def _resolve_run_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    config: dict | None,
    explicit_dests: set[str],
    config_user_keys: set[str] | None = None,
) -> argparse.Namespace:
    values = _parser_defaults(parser)
    user_keys = set()
    if config:
        values.update(config)
        user_keys.update(
            config_user_keys if config_user_keys is not None else config.keys()
        )

    cli_values = vars(args)
    for dest in explicit_dests:
        values[dest] = cli_values[dest]
    user_keys.update(explicit_dests)

    apply_t2v_lightning_defaults(values, user_keys)

    if values.get("no_lora"):
        values["lora"] = []
    if values.get("no_lora_high"):
        values["lora_high"] = []
    if values.get("no_lora_low"):
        values["lora_low"] = []

    return argparse.Namespace(**values)


def _expand_prompt_runs(
    run_args: argparse.Namespace, source: str = "CLI"
) -> list[argparse.Namespace]:
    prompt = run_args.prompt
    if prompt is None:
        raise SystemExit(f"{source}: --prompt is required")
    if isinstance(prompt, str):
        prompts = [prompt]
    elif isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
        prompts = prompt
    else:
        raise SystemExit(f"{source}: --prompt must be a string or list of strings")

    if not prompts or any(not item for item in prompts):
        raise SystemExit(f"{source}: --prompt is required")

    runs = []
    for prompt_index, prompt_text in enumerate(prompts):
        prompt_run = copy.copy(run_args)
        prompt_run.prompt = prompt_text
        prompt_run.prompt_index = prompt_index
        prompt_run.prompt_count = len(prompts)
        runs.append(prompt_run)
    return runs


def _validate_run_args(args: argparse.Namespace, source: str = "CLI") -> None:
    if not args.model_dir:
        raise SystemExit(f"{source}: --model-dir is required")
    if not args.prompt:
        raise SystemExit(f"{source}: --prompt is required")
    if args.iterations < 1:
        raise SystemExit(f"{source}: --iterations must be at least 1")

    for key, choices in CHOICES.items():
        value = getattr(args, key)
        if value not in choices:
            allowed = ", ".join(sorted(choices))
            raise SystemExit(f"{source}: {key} must be one of: {allowed}")


def _resolve_generation_runs(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    explicit_dests: set[str],
) -> list[argparse.Namespace]:
    if not args.config:
        run_args = _resolve_run_args(parser, args, None, explicit_dests)
        runs = _expand_prompt_runs(run_args)
        for run in runs:
            _validate_run_args(run)
        return runs

    runs = []
    for config_path in args.config:
        config, config_user_keys = _load_generation_config_with_defaults(config_path)
        run_args = _resolve_run_args(
            parser, args, config, explicit_dests, config_user_keys
        )
        prompt_runs = _expand_prompt_runs(run_args, str(config_path))
        for prompt_run in prompt_runs:
            _validate_run_args(prompt_run, str(config_path))
        runs.extend(prompt_runs)
    return runs


def _run_generation_args(args: argparse.Namespace) -> None:
    guide_scale = _parse_guide_scale(args.guide_scale)

    neg_prompt = args.negative_prompt
    if args.no_negative_prompt:
        neg_prompt = ""

    loras = _parse_lora_args(args.lora, "lora")
    loras_high = _parse_lora_args(args.lora_high, "lora_high")
    loras_low = _parse_lora_args(args.lora_low, "lora_low")
    output_dir = Path(args.output_path)
    prompt_count = getattr(args, "prompt_count", 1)
    prompt_index = getattr(args, "prompt_index", 0)
    uses_output_template = args.output_template is not None or prompt_count > 1
    output_path_is_dir = output_dir.is_dir()
    if args.iterations > 1 or uses_output_template:
        output_dir.mkdir(parents=True, exist_ok=True)

    wall_times = []
    mode = "i2v" if args.image else "t2v"
    if args.output_template:
        output_template = args.output_template
    elif prompt_count > 1:
        output_template = (
            "wan-{mode}-prompt{prompt_index1}-seed{seed}-s{steps}-sh{shift}-"
            "{width}x{height}.mp4"
        )
    else:
        output_template = "wan-{mode}-seed{seed}-s{steps}-sh{shift}-{width}x{height}.mp4"
    random_base_seed = args.seed
    if args.iteration_seed == "increment" and random_base_seed < 0:
        random_base_seed = random.randint(0, 2**32 - 1)

    for iteration in range(args.iterations):
        seed = _iteration_seed(random_base_seed, iteration, args.iteration_seed)
        if args.iterations == 1 and not uses_output_template and not output_path_is_dir:
            output_path = args.output_path
        else:
            try:
                output_path = str(
                    _iteration_output_path(
                        output_dir,
                        template=output_template,
                        mode=mode,
                        seed=seed,
                        steps=args.steps,
                        shift=args.shift,
                        width=args.width,
                        height=args.height,
                        frames=args.num_frames,
                        fps=args.fps,
                        iteration=iteration,
                        prompt_index=prompt_index,
                        prompt_count=prompt_count,
                    )
                )
            except OutputTemplateError as exc:
                raise SystemExit(f"output template error: {exc}") from exc
        print(f"[{iteration + 1}/{args.iterations}] seed={seed} output={output_path}")
        started = time.time()
        generate_video(
            model_dir=args.model_dir,
            prompt=args.prompt,
            negative_prompt=neg_prompt,
            image=args.image,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            steps=args.steps,
            guide_scale=guide_scale,
            shift=args.shift,
            refiner_start=args.refiner_start,
            seed=seed,
            output_path=output_path,
            fps=args.fps,
            scheduler=args.scheduler,
            sigma_schedule=args.sigma_schedule,
            noise_source=args.noise_source,
            torch_python=args.torch_python,
            t2v_lightning_preset=args.t2v_lightning_preset,
            positive_conditioning_npz=args.positive_conditioning_npz,
            negative_conditioning_npz=args.negative_conditioning_npz,
            dump_text_conditioning_npz=args.dump_text_conditioning_npz,
            dump_final_latents_npz=args.dump_final_latents_npz,
            initial_latents_npz=args.initial_latents_npz,
            loras=loras,
            loras_high=loras_high,
            loras_low=loras_low,
            tiling=args.tiling,
            legacy_vae_decode=args.legacy_vae_decode,
            no_compile=args.no_compile,
            trim_first_frames=args.trim_first_frames,
            debug_latents=args.debug_latents,
            output_last_frame=args.output_last_frame,
        )
        elapsed = time.time() - started
        wall_times.append(elapsed)
        print(f"[{iteration + 1}/{args.iterations}] wall_time={elapsed:.3f}s")

    if args.iterations > 1:
        print(f"Average wall time: {sum(wall_times) / len(wall_times):.3f}s")


def _torch_randn(shape: tuple[int, ...], seed: int, torch_python: str | None = None) -> mx.array:
    """Generate Comfy-compatible CPU Torch random normal noise."""
    if torch_python is None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "--noise-source torch requires torch in the active Python or --torch-python"
            ) from exc

        generator = torch.manual_seed(seed)
        arr = torch.randn(
            (1, *shape), dtype=torch.float32, device="cpu", generator=generator
        ).numpy()[0]
        return mx.array(arr.astype(np.float32, copy=False))

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
        tmp_path = Path(fh.name)
    try:
        script = (
            "import numpy as np, torch, sys; "
            "shape=tuple(int(x) for x in sys.argv[2].split(',')); "
            "g=torch.manual_seed(int(sys.argv[3])); "
            "arr=torch.randn((1,*shape), dtype=torch.float32, device='cpu', generator=g).numpy()[0]; "
            "np.save(sys.argv[1], arr)"
        )
        subprocess.run(
            [
                torch_python,
                "-c",
                script,
                str(tmp_path),
                ",".join(map(str, shape)),
                str(seed),
            ],
            check=True,
        )
        return mx.array(np.load(tmp_path).astype(np.float32, copy=False))
    finally:
        tmp_path.unlink(missing_ok=True)


def _load_text_conditioning_npz(path: str | Path) -> mx.array:
    """Load raw text conditioning with shape [tokens, dim]."""
    data = np.load(path)
    key = next((k for k in ("conditioning", "positive", "context") if k in data), None)
    if key is None:
        raise ValueError(
            f"{path} must contain one of: conditioning, positive, or context"
        )
    arr = data[key]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"{path}:{key} must have shape [tokens, dim] or [1, tokens, dim], got {arr.shape}"
        )
    return mx.array(arr.astype(np.float32, copy=False))


def _load_latents_npz(path: str | Path) -> mx.array:
    """Load model-space latents/noise with shape [C, T, H, W]."""
    data = np.load(path)
    key = next(
        (k for k in ("latents_model", "latents", "samples", "noise") if k in data),
        None,
    )
    if key is None:
        raise ValueError(
            f"{path} must contain latents_model, latents, samples, or noise"
        )
    arr = data[key]
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(
            f"{path}:{key} must have shape [C,T,H,W] or [1,C,T,H,W], got {arr.shape}"
        )
    return mx.array(arr.astype(np.float32, copy=False))


def _dump_text_conditioning_npz(
    path: str | Path,
    context: mx.array,
    context_null: mx.array | None,
) -> None:
    """Persist raw T5 embeddings before Wan model text projection."""
    payload = {"positive": np.array(context)}
    if context_null is not None:
        payload["negative"] = np.array(context_null)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _dump_latents_npz(path: str | Path, latents: mx.array, config) -> None:
    """Persist final latents in model space and VAE input layout."""
    payload = {"latents_model": np.array(latents)}
    if config.vae_z_dim == 48:
        from mlx_video.models.wan_2.vae22 import denormalize_latents

        z = latents.transpose(1, 2, 3, 0)[None]
        payload["latents_vae_thwc"] = np.array(denormalize_latents(z))
        payload["latents_vae_bcthw"] = payload["latents_vae_thwc"].transpose(
            0, 4, 1, 2, 3
        )
    else:
        payload["latents_vae_bcthw"] = np.array(latents[None])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _best_output_size(w, h, dw, dh, max_area):
    """Compute the best output resolution that fits within max_area while
    preserving the input aspect ratio and satisfying alignment constraints.
    Matches the reference implementation's best_output_size().
    """
    ratio = w / h
    ow = (max_area * ratio) ** 0.5
    oh = max_area / ow

    # Option 1: process width first
    ow1 = int(ow // dw * dw)
    oh1 = int(max_area / ow1 // dh * dh)
    ratio1 = ow1 / oh1

    # Option 2: process height first
    oh2 = int(oh // dh * dh)
    ow2 = int(max_area / oh2 // dw * dw)
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def generate_video(
    model_dir: str,
    prompt: str,
    negative_prompt: str | None = None,
    image: str | None = None,
    width: int = 1280,
    height: int = 704,
    num_frames: int = 81,
    steps: int = None,
    guide_scale: str | float | tuple = None,
    shift: float = None,
    refiner_start: float | int | None = None,
    seed: int = -1,
    output_path: str = "output.mp4",
    fps: int | None = None,
    scheduler: str = "unipc",
    sigma_schedule: str = "official",
    noise_source: str = "mlx",
    torch_python: str | None = None,
    t2v_lightning_preset: bool = False,
    positive_conditioning_npz: str | None = None,
    negative_conditioning_npz: str | None = None,
    dump_text_conditioning_npz: str | None = None,
    dump_final_latents_npz: str | None = None,
    initial_latents_npz: str | None = None,
    loras: list | None = None,
    loras_high: list | None = None,
    loras_low: list | None = None,
    tiling: str = "auto",
    legacy_vae_decode: bool = False,
    no_compile: bool = False,
    trim_first_frames: int = 0,
    debug_latents: bool = False,
    output_last_frame: bool | None = None,
):
    """Generate video using Wan pipeline (supports T2V and I2V).

    Args:
        model_dir: Path to converted MLX model directory
        prompt: Text prompt
        negative_prompt: Negative prompt (None = use config default, "" = no negative prompt)
        image: Path to input image for I2V (None = T2V mode)
        width: Video width
        height: Video height
        num_frames: Number of frames (must be 4n+1)
        steps: Number of diffusion steps (None = use config default)
        guide_scale: Guidance scale: float for single, (low,high) for dual (None = config default)
        shift: Noise schedule shift (None = use config default)
        refiner_start: Optional dual-model low-noise start point. Fractions in
            (0, 1) mean ceil(steps * value) high-noise steps; integer values
            use one-based low-noise start numbering.
        seed: Random seed (-1 for random)
        output_path: Output video path
        fps: Output frames per second (None = use config default)
        scheduler: Solver type: 'euler', 'dpm++', or 'unipc' (default)
        sigma_schedule: Sigma schedule: 'official' or 'comfy-simple'
        noise_source: Initial latent noise source: 'mlx' (default) or 'torch'
        torch_python: Optional Python executable used to generate Torch RNG noise
        t2v_lightning_preset: Apply Wan2.2 T2V Lightning reference defaults.
        positive_conditioning_npz: Optional raw positive text-conditioning NPZ
            to bypass MLX T5 encoding.
        negative_conditioning_npz: Optional raw negative text-conditioning NPZ.
        dump_text_conditioning_npz: Optional path to save raw MLX T5 conditioning.
        dump_final_latents_npz: Optional path to save final denoised latents.
        initial_latents_npz: Optional initial noise/latent tensor NPZ.
        loras: Optional list of (path, strength) tuples applied to all models
        loras_high: Optional list of (path, strength) tuples for high-noise model only
        loras_low: Optional list of (path, strength) tuples for low-noise model only
        tiling: Tiling mode for VAE decoding. Options:
            - "auto": Automatically determine tiling based on video size (default)
            - "none": Disable tiling
            - "default", "aggressive", "conservative": Preset tiling configs
            - "spatial": Spatial tiling only
            - "temporal": Temporal tiling only
        legacy_vae_decode: Preserve the old Wan2.1/14B single-frame VAE
            temporal upsample behavior for comparisons. Defaults to False,
            which uses reference single-frame decoding.
        no_compile: If True, skip mx.compile on models (useful for debugging)
        trim_first_frames: Number of temporal latent positions to generate extra
            and discard from the start. Each position = 4 pixel frames. Use 1
            to fix first-frame artifacts on 14B models (generates 4 extra frames,
            discards first 4). Use 2 for more aggressive trimming. Default: 0.
        debug_latents: If True, print per-temporal-position latent statistics
            after denoising for diagnosing first-frame artifacts.
        output_last_frame: Save the final decoded frame as a sibling PNG. If
            None, defaults to True when num_frames is 1.
    """
    from mlx_video.models.wan_2.config import WanModelConfig
    from mlx_video.models.wan_2.scheduler import (
        FlowDPMPP2MScheduler,
        FlowMatchEulerScheduler,
        FlowUniPCScheduler,
    )

    model_dir = Path(model_dir)

    # Load config from model dir if available, otherwise auto-detect
    config_path = model_dir / "config.json"
    quantization = None
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
        # Extract quantization config (not a model config field)
        quantization = config_dict.pop("quantization", None)
        # Handle tuple fields stored as lists in JSON
        for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
            if key in config_dict and isinstance(config_dict[key], list):
                config_dict[key] = tuple(config_dict[key])
        config = WanModelConfig(
            **{
                k: v
                for k, v in config_dict.items()
                if k in WanModelConfig.__dataclass_fields__
            }
        )
    else:
        # Auto-detect: dual model files → 2.2, single model → 2.1
        if (model_dir / "low_noise_model.safetensors").exists():
            config = WanModelConfig.wan22_t2v_14b()
        else:
            # Detect 1.3B vs 14B from weight shapes
            model_path = model_dir / "model.safetensors"
            if model_path.exists():
                probe = mx.load(str(model_path), return_metadata=False)
                for k, v in probe.items():
                    if "patch_embedding_proj.weight" in k:
                        dim = v.shape[0]
                        if dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                        break
                else:
                    config = WanModelConfig.wan21_t2v_14b()
                del probe
            else:
                config = WanModelConfig.wan21_t2v_14b()

    is_dual = config.dual_model
    is_i2v = image is not None
    if t2v_lightning_preset:
        if is_i2v:
            raise ValueError("--t2v-lightning-preset is only supported for T2V runs")
        parity_values = {
            "steps": steps,
            "guide_scale": guide_scale,
            "shift": shift,
            "scheduler": scheduler,
            "sigma_schedule": sigma_schedule,
            "fps": fps,
            "refiner_start": refiner_start,
            "noise_source": noise_source,
            "negative_prompt": negative_prompt,
            "lora_high": loras_high,
            "lora_low": loras_low,
        }
        user_keys = {
            key for key, value in parity_values.items() if value is not None
        }
        if noise_source == "mlx":
            user_keys.discard("noise_source")
        if scheduler == "unipc":
            user_keys.discard("scheduler")
        if sigma_schedule == "official":
            user_keys.discard("sigma_schedule")
        apply_t2v_lightning_defaults(parity_values, user_keys)
        steps = parity_values["steps"]
        guide_scale = parity_values["guide_scale"]
        shift = parity_values["shift"]
        scheduler = parity_values["scheduler"]
        sigma_schedule = parity_values["sigma_schedule"]
        fps = parity_values["fps"]
        refiner_start = parity_values["refiner_start"]
        noise_source = parity_values["noise_source"]
        negative_prompt = parity_values["negative_prompt"]
        loras_high = parity_values["lora_high"]
        loras_low = parity_values["lora_low"]

    # Validate config against actual weights (handles mismatched config.json)
    if not is_dual:
        model_path = model_dir / "model.safetensors"
        if model_path.exists():
            probe = mx.load(str(model_path), return_metadata=False)
            for k, v in probe.items():
                if "patch_embedding_proj.weight" in k:
                    actual_dim = v.shape[0]
                    if actual_dim != config.dim:
                        print(
                            f"{Colors.YELLOW}  Config dim={config.dim} doesn't match weights dim={actual_dim}, auto-correcting...{Colors.RESET}"
                        )
                        if actual_dim <= 2048:
                            config = WanModelConfig.wan21_t2v_1_3b()
                        else:
                            config = WanModelConfig.wan21_t2v_14b()
                    break
            del probe

    # Auto-correct Wan2.2 VAE params from stale configs
    if config.in_dim == 48 and config.vae_z_dim != 48:
        print(
            f"{Colors.YELLOW}  Auto-correcting Wan2.2 VAE params (in_dim=48 but vae_z_dim={config.vae_z_dim}){Colors.RESET}"
        )
        config = WanModelConfig(
            **{
                **{
                    f.name: getattr(config, f.name)
                    for f in config.__dataclass_fields__.values()
                },
                "vae_z_dim": 48,
                "vae_stride": (4, 16, 16),
                "sample_fps": 24,
            }
        )

    # Apply defaults from config if not overridden
    if steps is None:
        steps = config.sample_steps
    if shift is None:
        shift = config.sample_shift
    if guide_scale is None:
        guide_scale = config.sample_guide_scale
    output_fps = fps if fps is not None else config.sample_fps
    resolved_refiner_start = None
    high_noise_steps = None
    if refiner_start is not None:
        if not is_dual:
            raise ValueError("--refiner-start is only supported for dual-model Wan models")
        resolved_refiner_start, high_noise_steps = _resolve_refiner_start(
            refiner_start, steps
        )

    # Normalize guide_scale
    if isinstance(guide_scale, (int, float)):
        guide_scale = float(guide_scale)
    elif isinstance(guide_scale, str):
        parts = [float(x) for x in guide_scale.split(",")]
        guide_scale = tuple(parts) if len(parts) > 1 else parts[0]

    # Detect CFG-disabled mode (guide_scale=1.0 for all models → skip uncond pass for 2x speedup)
    if isinstance(guide_scale, tuple):
        cfg_disabled = all(gs <= 1.0 for gs in guide_scale)
    else:
        cfg_disabled = guide_scale <= 1.0

    # Validate frame count
    assert (num_frames - 1) % 4 == 0, f"num_frames must be 4n+1, got {num_frames}"

    gen_frames = num_frames
    if trim_first_frames > 0:
        gen_frames = num_frames + trim_first_frames * 4
        print(
            f"{Colors.DIM}  Trim: generating {gen_frames} frames, will discard first {trim_first_frames * 4}{Colors.RESET}"
        )

    version_str = f"Wan{config.model_version}"
    mode_str = "dual-model" if is_dual else "single-model"
    pipeline_str = "Image-to-Video" if is_i2v else "Text-to-Video"
    # Resolve negative prompt: explicit user value > config default
    # The official Wan2.2 uses a Chinese negative prompt (config.sample_neg_prompt)
    # that prevents oversaturation, artifacts, and comic look. We use it by default.
    # Text cleaning (_clean_text) normalizes fullwidth chars to match official tokenization.
    if negative_prompt is None:
        neg_prompt_resolved = config.sample_neg_prompt
    else:
        neg_prompt_resolved = negative_prompt
    print(f"{Colors.CYAN}{'='*60}")
    print(f"  {version_str} {pipeline_str} Generation (MLX, {mode_str})")
    print(f"{'='*60}{Colors.RESET}")
    print(f"{Colors.DIM}  Prompt: {prompt}")
    if is_i2v:
        print(f"  Image: {image}")
    if neg_prompt_resolved and neg_prompt_resolved.strip():
        neg_display = (
            neg_prompt_resolved[:60] + "..."
            if len(neg_prompt_resolved) > 60
            else neg_prompt_resolved
        )
        print(f"  Neg prompt: {neg_display}")
    print(f"  Size: {width}x{height}, Frames: {num_frames}, FPS: {output_fps}")
    print(
        f"  Steps: {steps}, Guide: {guide_scale}, Shift: {shift}, Solver: {scheduler}"
    )
    print(f"{Colors.DIM}  Sigma schedule: {sigma_schedule}{Colors.RESET}")
    print(f"{Colors.DIM}  Noise source: {noise_source}{Colors.RESET}")
    if config.vae_z_dim == 16:
        wan21_vae_decode = "legacy" if legacy_vae_decode else "reference"
        print(f"{Colors.DIM}  Wan2.1 VAE decode: {wan21_vae_decode}{Colors.RESET}")
    if resolved_refiner_start is not None:
        low_noise_steps = steps - high_noise_steps
        print(
            f"  Refiner start: step {resolved_refiner_start} "
            f"(HNE {high_noise_steps} step{'s' if high_noise_steps != 1 else ''}, "
            f"LNE {low_noise_steps} step{'s' if low_noise_steps != 1 else ''})"
        )
    if t2v_lightning_preset:
        print(f"  Reference sampling preset: enabled")
    if cfg_disabled:
        print(f"  CFG: disabled (guide_scale≤1 → B=1 fast path, 2x denoising speedup)")
    print(f"{Colors.RESET}")

    # Seed
    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    mx.random.seed(seed)
    np.random.seed(seed)
    print(f"{Colors.DIM}  Seed: {seed}{Colors.RESET}")

    # Align dimensions to patch_size * vae_stride (required for patchify)
    vae_stride = config.vae_stride
    patch_size = config.patch_size
    align_h = patch_size[1] * vae_stride[1]  # e.g. 2*16=32
    align_w = patch_size[2] * vae_stride[2]
    if height % align_h != 0 or width % align_w != 0:
        old_h, old_w = height, width
        height = (height // align_h) * align_h
        width = (width // align_w) * align_w
        if height == 0:
            height = align_h
        if width == 0:
            width = align_w
        print(
            f"{Colors.DIM}  Aligned {old_w}x{old_h} → {width}x{height} (must be divisible by {align_w}x{align_h}){Colors.RESET}"
        )

    # Enforce max_area constraint (model-specific resolution limit)
    if config.max_area > 0 and height * width > config.max_area:
        old_h, old_w = height, width
        width, height = _best_output_size(
            width, height, align_w, align_h, config.max_area
        )
        print(
            f"{Colors.YELLOW}  ⚠ Resolution {old_w}x{old_h} exceeds model's max area "
            f"({config.max_area:,}px). Adjusted → {width}x{height}{Colors.RESET}"
        )

    # Compute target latent shape
    z_dim = config.vae_z_dim
    t_latent = (gen_frames - 1) // vae_stride[0] + 1
    h_latent = height // vae_stride[1]
    w_latent = width // vae_stride[2]
    target_shape = (z_dim, t_latent, h_latent, w_latent)

    # Sequence length for transformer
    seq_len = math.ceil(
        (h_latent * w_latent) / (patch_size[1] * patch_size[2]) * t_latent
    )

    print(f"{Colors.DIM}  Latent shape: {target_shape}")
    print(f"  Sequence length: {seq_len}{Colors.RESET}")

    # Load or compute raw text conditioning.
    t1 = time.time()
    if positive_conditioning_npz:
        print(f"\n{Colors.BLUE}Loading text conditioning...{Colors.RESET}")
        context = _load_text_conditioning_npz(positive_conditioning_npz)
        if cfg_disabled:
            context_null = None
            mx.eval(context)
        elif negative_conditioning_npz:
            context_null = _load_text_conditioning_npz(negative_conditioning_npz)
            mx.eval(context, context_null)
        else:
            raise ValueError(
                "--negative-conditioning-npz is required when CFG is enabled"
            )
        print(
            f"{Colors.DIM}  Positive conditioning: {positive_conditioning_npz} shape={context.shape}"
        )
        if context_null is not None:
            print(
                f"  Negative conditioning: {negative_conditioning_npz} shape={context_null.shape}"
            )
        print(f"  Text conditioning load: {time.time() - t1:.1f}s{Colors.RESET}")
    else:
        print(f"\n{Colors.BLUE}Loading T5 encoder...{Colors.RESET}")
        t5_encoder = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)

        # Load tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

        # Encode prompts
        print(f"{Colors.BLUE}Encoding text...{Colors.RESET}")
        context = encode_text(t5_encoder, tokenizer, prompt, config.text_len)
        if cfg_disabled:
            context_null = None
            mx.eval(context)
        else:
            context_null = encode_text(
                t5_encoder,
                tokenizer,
                neg_prompt_resolved,
                config.text_len,
            )
            mx.eval(context, context_null)

        if dump_text_conditioning_npz:
            _dump_text_conditioning_npz(
                dump_text_conditioning_npz, context, context_null
            )
            print(
                f"{Colors.DIM}  Text conditioning dumped to {dump_text_conditioning_npz}{Colors.RESET}"
            )

        # Free T5 from memory
        del t5_encoder
        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  T5 encoding: {time.time() - t1:.1f}s{Colors.RESET}")

    # I2V: encode image to latent space
    z_img = None
    i2v_mask = None
    i2v_mask_tokens = None
    y_i2v = None
    is_i2v_channel_concat = is_i2v and config.model_type == "i2v"
    is_i2v_mask_blend = is_i2v and config.model_type != "i2v"
    if is_i2v:
        print(f"\n{Colors.BLUE}Encoding input image...{Colors.RESET}")
        t_img = time.time()

        vae_path = model_dir / "vae.safetensors"

        if is_i2v_channel_concat:
            # I2V-14B: encode full video (first frame = image, rest = zeros)
            # and construct y tensor with mask + encoded latents
            from PIL import Image

            img = Image.open(image).convert("RGB")
            scale = max(width / img.width, height / img.height)
            img = img.resize(
                (round(img.width * scale), round(img.height * scale)), Image.LANCZOS
            )
            x1, y1 = (img.width - width) // 2, (img.height - height) // 2
            img = img.crop((x1, y1, x1 + width, y1 + height))
            img_arr = mx.array(
                np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0
            )  # [H, W, 3]
            img_chw = img_arr.transpose(2, 0, 1)  # [3, H, W]

            # Build video: first frame = image, rest = zeros -> [3, F, H, W]
            # Chunked encoding processes 1-frame + 4-frame chunks with temporal caching
            video = mx.concatenate(
                [
                    img_chw[:, None, :, :],
                    mx.zeros((3, num_frames - 1, height, width)),
                ],
                axis=1,
            )

            # Encode through Wan2.1 VAE -> [1, z_dim, T_lat, H_lat, W_lat]
            vae_enc = load_vae_encoder(vae_path, config)
            z_video = vae_enc.encode(video[None])  # [1, 16, T_lat, H_lat, W_lat]
            mx.eval(z_video)
            z_video = z_video[0]  # [16, T_lat, H_lat, W_lat]

            # Build mask: 1 for first frame, 0 for rest -> rearrange to [4, T_lat, H, W]
            msk = mx.ones((1, num_frames, h_latent, w_latent))
            msk = mx.concatenate(
                [msk[:, :1], mx.zeros((1, num_frames - 1, h_latent, w_latent))], axis=1
            )
            # Repeat first frame 4x, concat rest: [1, 4 + (F-1), H_lat, W_lat]
            msk = mx.concatenate(
                [
                    mx.repeat(msk[:, :1], 4, axis=1),
                    msk[:, 1:],
                ],
                axis=1,
            )
            # Reshape to [1, T_lat, 4, H_lat, W_lat] then transpose -> [4, T_lat, H_lat, W_lat]
            msk = msk.reshape(1, msk.shape[1] // 4, 4, h_latent, w_latent)
            msk = msk.transpose(0, 2, 1, 3, 4)[0]  # [4, T_lat, H_lat, W_lat]

            # y = concat([mask, encoded_video]) -> [20, T_lat, H_lat, W_lat]
            y_i2v = mx.concatenate([msk, z_video], axis=0)
            mx.eval(y_i2v)

            del vae_enc, img_arr, img_chw, video, z_video, msk
        else:
            # TI2V-5B: encode single image, blend with noise via mask
            img_tensor = preprocess_image(image, width, height)
            mx.eval(img_tensor)

            vae_enc = load_vae_encoder(vae_path, config)
            z_img = vae_enc.encode(img_tensor)  # [1, 1, H_lat, W_lat, z_dim]
            mx.eval(z_img)
            z_img = z_img[0].transpose(3, 0, 1, 2)  # [z_dim, 1, H_lat, W_lat]
            i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, config.patch_size)

            del vae_enc, img_tensor

        gc.collect()
        mx.clear_cache()
        print(f"{Colors.DIM}  Image encoding: {time.time() - t_img:.1f}s{Colors.RESET}")

    # Load transformer models
    print(f"\n{Colors.BLUE}Loading transformer model(s)...{Colors.RESET}")
    if quantization:
        print(
            f"{Colors.DIM}  Using {quantization['bits']}-bit quantized weights (group_size={quantization['group_size']}){Colors.RESET}"
        )
    t2 = time.time()

    # Merge per-model LoRAs with shared LoRAs
    _loras_low = (loras or []) + (loras_low or []) or None
    _loras_high = (loras or []) + (loras_high or []) or None
    _loras_single = loras

    if is_dual:
        low_noise_path = model_dir / "low_noise_model.safetensors"
        high_noise_path = model_dir / "high_noise_model.safetensors"
        low_noise_model = load_wan_model(
            low_noise_path, config, quantization, loras=_loras_low
        )
        high_noise_model = load_wan_model(
            high_noise_path, config, quantization, loras=_loras_high
        )
    else:
        single_model = load_wan_model(
            model_dir / "model.safetensors", config, quantization, loras=_loras_single
        )
    print(f"{Colors.DIM}  Models loaded: {time.time() - t2:.1f}s{Colors.RESET}")

    # Precompute text embeddings once (avoids redundant MLP in every step)
    # Each model has its own text_embedding weights, so dual models need separate embeddings
    if cfg_disabled:
        # No CFG: only compute cond embeddings (B=1 forward pass, 2x faster)
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context])
            context_emb_high = high_noise_model.embed_text([context])
            mx.eval(context_emb_low, context_emb_high)
            context_cond_low = context_emb_low[0:1]
            context_cond_high = context_emb_high[0:1]
        else:
            context_emb = single_model.embed_text([context])
            mx.eval(context_emb)
            context_cond = context_emb[0:1]
    else:
        if is_dual:
            context_emb_low = low_noise_model.embed_text([context, context_null])
            context_emb_high = high_noise_model.embed_text([context, context_null])
            mx.eval(context_emb_low, context_emb_high)
            context_cfg_low = mx.concatenate(
                [context_emb_low[0:1], context_emb_low[1:2]], axis=0
            )
            context_cfg_high = mx.concatenate(
                [context_emb_high[0:1], context_emb_high[1:2]], axis=0
            )
        else:
            context_emb = single_model.embed_text([context, context_null])
            mx.eval(context_emb)
            context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)

    # Precompute cross-attention K/V caches (constant across all steps)
    if cfg_disabled:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cond_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cond_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cond)
            mx.eval(cross_kv)
    else:
        if is_dual:
            cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
            cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
            mx.eval(cross_kv_low, cross_kv_high)
        else:
            cross_kv = single_model.prepare_cross_kv(context_cfg)
            mx.eval(cross_kv)

    # Precompute RoPE frequencies (grid sizes are constant across all steps)
    f_grid = t_latent // patch_size[0]
    h_grid = h_latent // patch_size[1]
    w_grid = w_latent // patch_size[2]
    if cfg_disabled:
        rope_grid_sizes = [(f_grid, h_grid, w_grid)]
    else:
        rope_grid_sizes = [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    if is_dual:
        rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
        rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    else:
        rope_cos_sin = single_model.prepare_rope(rope_grid_sizes)
        mx.eval(rope_cos_sin)

    # Setup scheduler
    _schedulers = {
        "euler": FlowMatchEulerScheduler,
        "dpm++": FlowDPMPP2MScheduler,
        "unipc": FlowUniPCScheduler,
    }
    sched_cls = _schedulers.get(scheduler, FlowUniPCScheduler)
    if scheduler == "euler":
        sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    else:
        sched = sched_cls(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(steps, shift=shift, sigma_schedule=sigma_schedule)

    # Generate or load initial noise
    if initial_latents_npz:
        noise = _load_latents_npz(initial_latents_npz)
        if tuple(noise.shape) != tuple(target_shape):
            raise ValueError(
                f"--initial-latents-npz shape {noise.shape} does not match target {target_shape}"
            )
        print(
            f"{Colors.DIM}  Initial latents loaded from {initial_latents_npz}{Colors.RESET}"
        )
    elif noise_source == "mlx":
        noise = mx.random.normal(target_shape)
    elif noise_source == "torch":
        noise = _torch_randn(target_shape, seed, torch_python=torch_python)
        print(f"{Colors.DIM}  Initial latents generated with Torch RNG{Colors.RESET}")
    else:
        raise ValueError(f"Unsupported noise source: {noise_source}")

    # I2V initialization: TI2V-5B blends image with noise, I2V-14B uses pure noise
    if is_i2v_mask_blend:
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    else:
        latents = noise

    # Boundary for model switching (dual model only)
    boundary = (config.boundary * config.num_train_timesteps) if is_dual else None

    # Diffusion loop
    print(f"\n{Colors.GREEN}Denoising ({steps} steps)...{Colors.RESET}")
    t3 = time.time()

    # Compile model forward for faster denoising
    if not no_compile:
        models_to_compile = (
            [high_noise_model, low_noise_model] if is_dual else [single_model]
        )
        for m in models_to_compile:
            m._compiled = mx.compile(m)

    # Pre-convert timesteps to Python list to avoid .item() sync each step
    timestep_list = sched.timesteps.tolist()

    for i, t in enumerate(tqdm(range(steps), desc="Diffusion")):
        timestep_val = timestep_list[i]

        # Select model, cached K/V, and precomputed RoPE
        if is_dual:
            use_high_noise = (
                i < high_noise_steps
                if high_noise_steps is not None
                else timestep_val >= boundary
            )
            if use_high_noise:
                model = high_noise_model
                kv = cross_kv_high
                rcs = rope_cos_sin_high
            else:
                model = low_noise_model
                kv = cross_kv_low
                rcs = rope_cos_sin_low
        else:
            model = single_model
            kv = cross_kv
            rcs = rope_cos_sin

        # Use compiled forward when available (faster after first trace)
        _call = getattr(model, "_compiled", model)

        if cfg_disabled:
            # No CFG: B=1 forward pass (2x faster than B=2 CFG batch)
            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = t_tokens  # [1, L]
            else:
                t_batch = mx.array([timestep_val])

            y_arg = [y_i2v] if is_i2v_channel_concat else None

            if is_dual:
                ctx = context_cond_high if use_high_noise else context_cond_low
            else:
                ctx = context_cond
            preds = _call(
                [latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred = preds[0]
            del preds
        else:
            # CFG: batch cond + uncond into single B=2 forward pass
            if is_dual:
                gs = guide_scale[1] if use_high_noise else guide_scale[0]
            else:
                gs = (
                    guide_scale
                    if isinstance(guide_scale, (int, float))
                    else guide_scale[0]
                )

            if is_i2v_mask_blend:
                t_tokens = i2v_mask_tokens * timestep_val
                pad_len = seq_len - t_tokens.shape[1]
                if pad_len > 0:
                    t_tokens = mx.concatenate(
                        [t_tokens, mx.full((1, pad_len), timestep_val)], axis=1
                    )
                t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)
            else:
                t_batch = mx.array([timestep_val, timestep_val])

            y_arg = [y_i2v, y_i2v] if is_i2v_channel_concat else None

            ctx = (
                context_cfg
                if not is_dual
                else (context_cfg_high if use_high_noise else context_cfg_low)
            )
            preds = _call(
                [latents, latents],
                t=t_batch,
                context=ctx,
                seq_len=seq_len,
                cross_kv_caches=kv,
                y=y_arg,
                rope_cos_sin=rcs,
            )
            noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
            noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)
            del noise_pred_cond, noise_pred_uncond, preds

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)

        # TI2V-5B: re-apply mask to keep first frame frozen
        if is_i2v_mask_blend:
            latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents

        # Release temporaries before eval to free memory for graph execution
        del noise_pred
        mx.eval(latents)

    print(f"{Colors.DIM}  Denoising: {time.time() - t3:.1f}s{Colors.RESET}")

    if dump_final_latents_npz:
        _dump_latents_npz(dump_final_latents_npz, latents, config)
        print(
            f"{Colors.DIM}  Final latents dumped to {dump_final_latents_npz}{Colors.RESET}"
        )

    # Diagnostic: per-temporal-position latent statistics
    if debug_latents:
        lat_np = np.array(latents)  # [C, T, H, W]
        n_t = lat_np.shape[1]
        print(
            f"\n{Colors.CYAN}  Latent diagnostics (shape {lat_np.shape}):{Colors.RESET}"
        )
        print(
            f"  {'Pos':>4s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}  {'AbsMean':>8s}"
        )
        for t_pos in range(min(n_t, 8)):
            frame = lat_np[:, t_pos, :, :]
            print(
                f"  {t_pos:4d}  {frame.mean():8.4f}  {frame.std():8.4f}  "
                f"{frame.min():8.4f}  {frame.max():8.4f}  {np.abs(frame).mean():8.4f}"
            )
        if n_t > 8:
            interior = lat_np[:, 4:, :, :]
            print(
                f"  {'4+':>4s}  {interior.mean():8.4f}  {interior.std():8.4f}  "
                f"{interior.min():8.4f}  {interior.max():8.4f}  {np.abs(interior).mean():8.4f}"
            )
        print()

    # Free transformer models and text embeddings
    if is_dual:
        del low_noise_model, high_noise_model, cross_kv_low, cross_kv_high
        if cfg_disabled:
            del context_cond_low, context_cond_high
        else:
            del context_cfg_low, context_cfg_high
    else:
        del single_model, cross_kv
        if cfg_disabled:
            del context_cond
        else:
            del context_cfg
    del model, kv, context
    if context_null is not None:
        del context_null
    gc.collect()
    mx.clear_cache()

    # Load VAE and decode
    print(f"\n{Colors.BLUE}Decoding with VAE...{Colors.RESET}")
    t4 = time.time()
    vae_path = model_dir / "vae.safetensors"
    vae = load_vae_decoder(vae_path, config)

    is_wan22_vae = config.vae_z_dim == 48

    # Temporal extend: prepend reflected latent frames to the VAE input so that
    # the CausalConv3d zero-padding artifacts fall on the prefix (which we crop).
    # This gives the first real frame a full temporal receptive field of real data.
    # Select tiling configuration
    from mlx_video.models.ltx_2.video_vae.tiling import TilingConfig

    if tiling == "none":
        tiling_config = None
    elif tiling == "auto":
        tiling_config = TilingConfig.auto(height, width, num_frames)
    elif tiling == "default":
        tiling_config = TilingConfig.default()
    elif tiling == "aggressive":
        tiling_config = TilingConfig.aggressive()
    elif tiling == "conservative":
        tiling_config = TilingConfig.conservative()
    elif tiling == "spatial":
        tiling_config = TilingConfig.spatial_only()
    elif tiling == "temporal":
        tiling_config = TilingConfig.temporal_only()
    else:
        print(
            f"{Colors.YELLOW}  Unknown tiling mode '{tiling}', using auto{Colors.RESET}"
        )
        tiling_config = TilingConfig.auto(height, width, num_frames)

    if tiling_config is not None:
        spatial_info = (
            f"{tiling_config.spatial_config.tile_size_in_pixels}px"
            if tiling_config.spatial_config
            else "none"
        )
        temporal_info = (
            f"{tiling_config.temporal_config.tile_size_in_frames}f"
            if tiling_config.temporal_config
            else "none"
        )
        print(
            f"{Colors.DIM}  Tiling ({tiling}): spatial={spatial_info}, temporal={temporal_info}{Colors.RESET}"
        )

    if is_wan22_vae:
        from mlx_video.models.wan_2.vae22 import denormalize_latents

        # latents: [C, T, H, W] → [1, T, H, W, C] (channels-last for Wan2.2 VAE)
        z = latents.transpose(1, 2, 3, 0)[None]
        z = denormalize_latents(z)
        if tiling_config is not None:
            video = vae.decode_tiled(z, tiling_config)
        else:
            video = vae(z)
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [T', H', W', 3]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    else:
        if (
            not legacy_vae_decode
            and tiling_config is not None
            and latents.shape[1] == 1
        ):
            tiling_config = None
            print(
                f"{Colors.DIM}  Tiling disabled for single-frame Wan2.1 VAE reference decode{Colors.RESET}"
            )
        if tiling_config is not None:
            video = vae.decode_tiled(latents[None], tiling_config)
        else:
            wan21_vae_decode = "legacy" if legacy_vae_decode else "reference"
            video = vae.decode(latents[None], decode_mode=wan21_vae_decode)
        mx.eval(video)
        print(f"{Colors.DIM}  VAE decode: {time.time() - t4:.1f}s{Colors.RESET}")

        video = np.array(video[0])  # [3, T', H, W]
        video = (video + 1.0) / 2.0
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
        video = video.transpose(1, 2, 3, 0)  # [T, H, W, 3]

    # Trim first N temporal chunks if requested (avoids first-frame artifacts)
    if trim_first_frames > 0:
        trim_pixels = trim_first_frames * 4
        video = video[trim_pixels:]
        print(
            f"{Colors.DIM}  Trimmed first {trim_pixels} frames ({video.shape[0]} remaining){Colors.RESET}"
        )

    decoded_frames = video.shape[0]
    video = _crop_decoded_video(video, num_frames)
    if decoded_frames != video.shape[0]:
        print(
            f"{Colors.DIM}  Decoded frames: {decoded_frames}; output frames: {video.shape[0]}{Colors.RESET}"
        )

    if should_output_last_frame(output_last_frame, num_frames):
        try:
            png_path = save_last_frame_png(video, output_path)
            print(f"{Colors.GREEN}✓ Last frame saved to {png_path}{Colors.RESET}")
        except OSError as exc:
            print(
                f"{Colors.YELLOW}  Could not save last-frame PNG for {output_path}: {exc}{Colors.RESET}"
            )

    save_video(video, output_path, fps=output_fps)
    print(f"\n{Colors.GREEN}✓ Video saved to {output_path}{Colors.RESET}")
    print(f"{Colors.DIM}  Total time: {time.time() - t1:.1f}s{Colors.RESET}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wan Text-to-Video Generation (MLX)")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help=(
            "Path to a JSON or YAML generation config. Repeat to run multiple "
            "configs sequentially; explicit CLI flags override config values."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to converted MLX model directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help="Text prompt. Repeat to generate once per prompt.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for I2V (omit for T2V mode)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt for CFG (default: official Chinese prompt from config)",
    )
    parser.add_argument(
        "--no-negative-prompt",
        action="store_true",
        help="Disable negative prompt (use empty string instead of config default)",
    )
    parser.add_argument(
        "--width", type=int, default=1280, help="Video width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=704,
        help="Video height (default: 704; 720p models use 704)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=81, help="Number of frames (must be 4n+1)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of diffusion steps (default: from config)",
    )
    parser.add_argument(
        "--guide-scale",
        type=str,
        default=None,
        help="Guidance scale: single float or low,high pair",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=None,
        help="Noise schedule shift (default: from config)",
    )
    parser.add_argument(
        "--refiner-start",
        type=float,
        default=None,
        help=(
            "Dual-model low-noise start. Fractions in (0, 1) use "
            "ceil(steps*value) high-noise steps; integers 1..steps+1 are "
            "one-based low-noise start steps."
        ),
    )
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument(
        "--output-path", type=str, default="output.mp4", help="Output video path"
    )
    parser.add_argument("--fps", type=int, default=None, help="Output frames per second")
    parser.add_argument(
        "--output-last-frame",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save the final decoded frame as a PNG beside the MP4 "
            "(default: enabled when --num-frames is 1)"
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["euler", "dpm++", "unipc"],
        help="Diffusion solver: euler (1st order), dpm++ (2nd order), unipc (2nd order PC, default/official)",
    )
    parser.add_argument(
        "--sigma-schedule",
        type=str,
        default="official",
        choices=["official", "comfy-simple"],
        help="Sigma schedule: official Wan/MLX interpolation or ComfyUI ModelSamplingSD3 simple",
    )
    parser.add_argument(
        "--noise-source",
        choices=["mlx", "torch"],
        default="mlx",
        help=(
            "Initial latent noise source. Use 'torch' for PyTorch-compatible "
            "seed parity checks; default: mlx"
        ),
    )
    parser.add_argument(
        "--torch-python",
        default=None,
        help=(
            "Python executable used when --noise-source torch should generate "
            "noise outside the active environment"
        ),
    )
    parser.add_argument(
        "--t2v-lightning-preset",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--positive-conditioning-npz",
        type=str,
        default=None,
        help=(
            "Load raw positive text-conditioning embeddings from an NPZ file "
            "and bypass MLX T5 encoding (diagnostic)"
        ),
    )
    parser.add_argument(
        "--negative-conditioning-npz",
        type=str,
        default=None,
        help=(
            "Load raw negative text-conditioning embeddings from an NPZ file "
            "(required with --positive-conditioning-npz when CFG is enabled)"
        ),
    )
    parser.add_argument(
        "--dump-text-conditioning-npz",
        type=str,
        default=None,
        help="Save MLX raw T5 text-conditioning embeddings to an NPZ file",
    )
    parser.add_argument(
        "--dump-final-latents-npz",
        type=str,
        default=None,
        help="Save final denoised latents to an NPZ file before VAE decode",
    )
    parser.add_argument(
        "--initial-latents-npz",
        type=str,
        default=None,
        help="Load initial noise/latents from an NPZ file instead of RNG",
    )
    parser.add_argument(
        "--lora",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to all models (repeatable). Format: --lora path.safetensors 0.8",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Clear generic LoRAs inherited from config",
    )
    parser.add_argument(
        "--lora-high",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to high-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--no-lora-high",
        action="store_true",
        help="Clear high-noise LoRAs inherited from config",
    )
    parser.add_argument(
        "--lora-low",
        nargs=2,
        action="append",
        metavar=("PATH", "STRENGTH"),
        help="Apply a LoRA to low-noise model only (dual-model, repeatable)",
    )
    parser.add_argument(
        "--no-lora-low",
        action="store_true",
        help="Clear low-noise LoRAs inherited from config",
    )
    parser.add_argument(
        "--tiling",
        type=str,
        default="auto",
        choices=[
            "auto",
            "none",
            "default",
            "aggressive",
            "conservative",
            "spatial",
            "temporal",
        ],
        help="VAE tiling mode to reduce memory during decoding (default: auto)",
    )
    parser.add_argument(
        "--legacy-vae-decode",
        action="store_true",
        help=(
            "Use the old Wan2.1/14B single-frame VAE temporal upsample path "
            "instead of the reference single-frame decode"
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable mx.compile on models (for debugging)",
    )
    parser.add_argument(
        "--trim-first-frames",
        type=int,
        default=0,
        metavar="N",
        help="Generate N extra temporal chunks (N×4 frames) and discard them from the start. "
        "Fixes first-frame color/lighting artifacts on 14B models. Try 1 first (4 frames). "
        "Default: 0 (disabled)",
    )
    parser.add_argument(
        "--debug-latents",
        action="store_true",
        help="Print per-temporal-position latent statistics after denoising (diagnostic)",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--iteration-seed",
        choices=["same", "increment", "random"],
        default="increment",
        help="Seed strategy for subsequent iterations",
    )
    parser.add_argument(
        "--output-template",
        default=None,
        help="Relative output filename template used under --output-path",
    )
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    if argv is None:
        import sys

        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    explicit_dests = _explicit_cli_dests(parser, argv)
    runs = _resolve_generation_runs(parser, args, explicit_dests)
    for run_args in runs:
        _run_generation_args(run_args)


if __name__ == "__main__":
    main()

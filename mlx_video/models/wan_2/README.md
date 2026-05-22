
## Wan2.1 / Wan2.2

Both [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2) are text-to-video diffusion models built on a DiT (Diffusion Transformer) backbone with a T5 text encoder and 3D VAE. 

They share the same model architecture — the difference is in the inference pipeline:

| | Wan2.1 | Wan2.2 T2V-14B | Wan2.2 I2V-14B | Wan2.2 TI2V-5B |
|---|--------|--------|--------|--------|
| **Task** | Text-to-Video | Text-to-Video | Image-to-Video | Text+Image-to-Video |
| **Pipeline** | Single model | Dual model | Dual model | Single model |
| **Sizes** | 1.3B, 14B | 14B | 14B | 5B |
| **Resolution** | 480P (1.3B), 720P (14B) | 720P | 720P | 720P |
| **Steps** | 50 | 40 | 40 | 40 |
| **Guidance** | 5.0 (fixed) | 3.0 / 4.0 | 3.5 / 3.5 | 5.0 (fixed) |
| **Shift** | 5.0 | 12.0 | 5.0 | 5.0 |
| **VAE** | Wan2.1 (z=16) | Wan2.1 (z=16) | Wan2.1 (z=16) + encoder | Wan2.2 (z=48) |

### Step 1: Download Weights

Download the original PyTorch checkpoints from HuggingFace using the `huggingface-cli` tool (install with `pip install huggingface_hub`):

**Wan2.1**
```bash
# Text-to-Video 1.3B (fast, fits in ~4 GB)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# Text-to-Video 14B
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

**Wan2.2**
```bash
# Text-to-Video 14B
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B --local-dir ./Wan2.2-T2V-A14B

# Image-to-Video 14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Wan2.2-I2V-A14B

# Text+Image-to-Video 5B (uses a different VAE — z_dim=48)
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./Wan2.2-TI2V-5B
```

Each downloaded directory will have this structure:

```
Wan2.1-T2V-*/
├── models_t5_umt5-xxl-enc-bf16.pth       # T5 text encoder
├── Wan2.1_VAE.pth                         # 3D VAE
└── diffusion_pytorch_model*.safetensors   # transformer (single)

Wan2.2-T2V-A14B/ or Wan2.2-I2V-A14B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
├── low_noise_model/                       # dual-model low-noise transformer
└── high_noise_model/                      # dual-model high-noise transformer

Wan2.2-TI2V-5B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.2_VAE.pth                         # different VAE (z_dim=48)
└── diffusion_pytorch_model*.safetensors   # transformer (single)
```

> **Wan2.2 I2V-14B** shares the same directory structure as Wan2.2 T2V. The conversion script auto-detects I2V from the model's `config.json` (`model_type: "i2v"`, `in_dim: 36`).

### Step 2: Convert to MLX Format

The conversion script auto-detects the model version from the directory structure (presence of `low_noise_model/` → Wan2.2 dual model) and the model type from `config.json` (I2V vs T2V).

#### Wan2.1 T2V 1.3B

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.1-T2V-1.3B \
    --output-dir ./Wan2.1-T2V-1.3B-MLX
```

#### Wan2.1 T2V 14B

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.1-T2V-14B \
    --output-dir ./Wan2.1-T2V-14B-MLX
```

#### Wan2.2 T2V 14B

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-T2V-A14B \
    --output-dir ./Wan2.2-T2V-A14B-MLX
```

#### Wan2.2 I2V 14B

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-I2V-A14B \
    --output-dir ./Wan2.2-I2V-A14B-MLX
```

The I2V model is auto-detected from `config.json`; the output will include a `vae_encoder.safetensors` used to encode the conditioning image.

#### Wan2.2 TI2V 5B

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-TI2V-5B \
    --output-dir ./Wan2.2-TI2V-5B-MLX
```

The TI2V model uses a different VAE (`z_dim=48`, `vae_stride=(4,16,16)`) and is auto-detected during conversion.

---

You can also pass `--model-version 2.1` or `--model-version 2.2` to force the version instead of relying on auto-detection.

#### Conversion Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint-dir` | (required) | Path to original PyTorch checkpoint directory |
| `--output-dir` | `wan_mlx_model` | Output path for MLX model |
| `--dtype` | `bfloat16` | Target dtype (`float16`, `float32`, `bfloat16`) |
| `--model-version` | `auto` | Model version: `2.1`, `2.2`, or `auto` |
| `--quantize` | off | Quantize transformer weights for reduced memory |
| `--bits` | `4` | Quantization bits: `4` or `8` |
| `--group-size` | `64` | Quantization group size: `32`, `64`, or `128` |

The converter produces:
```
wan_mlx/
├── config.json                    # Model configuration
├── t5_encoder.safetensors         # T5 UMT5-XXL text encoder
├── vae.safetensors                # 3D VAE decoder
├── vae_encoder.safetensors        # 3D VAE encoder (I2V-14B only)
├── model.safetensors              # (Wan2.1) Single transformer
├── low_noise_model.safetensors    # (Wan2.2) Low-noise transformer
└── high_noise_model.safetensors   # (Wan2.2) High-noise transformer
```

### Step 3: Generate Video

#### Wan2.1 T2V 1.3B

```bash
python -m mlx_video.wan2.gemer \
    --model-dir ./Wan2.1-T2V-1.3B-MLX \
    --prompt "A cat playing piano in a cozy living room, cinematic lighting" \
    --width 832 --height 480 --num-frames 81 \
    --steps 50 --guide-scale 5.0 \
    --seed 42 \
    --output-path wan21_1b.mp4
```

#### Wan2.1 T2V 14B

```bash
python -m mlx_video.wan2.gemer \
    --model-dir ./Wan2.1-T2V-14B-MLX \
    --prompt "A woman walks through a misty forest at dawn, slow motion, cinematic" \
    --width 1280 --height 704 --num-frames 81 \
    --steps 50 --guide-scale 5.0 \
    --seed 42 \
    --output-path wan21_14b.mp4
```

> **Tip**: If the first few frames look washed out or have color artifacts, add `--trim-first-frames 1` to generate 4 extra frames at the start and discard them. With the `unipc` scheduler (default), **10 steps** often gives satisfying results — useful for quick iteration.

#### Wan2.2 T2V 14B

Wan2.2 uses a dual-model pipeline (separate high-noise and low-noise transformers) and takes guidance as a `high,low` pair:

```bash
python -m mlx_video.wan2.generate \
    --model-dir ./Wan2.2-T2V-A14B-MLX \
    --prompt "Two astronauts playing chess on the surface of the moon, dramatic lighting, 8K" \
    --negative-prompt "low quality, blurry, distorted" \
    --width 1280 --height 704 --num-frames 81 \
    --steps 40 --guide-scale "3.0,4.0" \
    --seed 42 \
    --output-path wan22_t2v.mp4
```

> **Tip**: With the `unipc` scheduler (default), **10 steps** often produces satisfying results for 14B models — a significant speed-up with minimal quality loss. Try `--steps 10` for quick iterations.

#### Wan2.2 I2V 14B

Image-to-video: animates a starting image guided by a text prompt. Pass the image with `--image`:

```bash
python -m mlx_video.wan2.generate \
    --model-dir ./Wan2.2-I2V-A14B-MLX \
    --image ./my_photo.png \
    --prompt "The person slowly turns their head and smiles, cinematic, natural lighting" \
    --negative-prompt "low quality, blurry, distorted" \
    --width 1280 --height 704 --num-frames 81 \
    --steps 40 --guide-scale "3.5,3.5" \
    --seed 42 \
    --output-path wan22_i2v.mp4
```

> **Tip**: As with T2V, `--steps 10` with the `unipc` scheduler is often sufficient for fast prototyping.

#### Wan2.2 TI2V 5B

Text+image-to-video: a single-model variant with a larger VAE (`z_dim=48`). Resolution must be divisible by **32** (not 16 as with other models):

```bash
python -m mlx_video.wan2.generate \
    --model-dir ./Wan2.2-TI2V-5B-MLX \
    --image ./my_photo.png \
    --prompt "The subject waves hello, warm sunlight, film grain" \
    --width 1280 --height 704 --num-frames 41 \
    --steps 40 --guide-scale 5.0 \
    --seed 42 \
    --output-path wan22_ti2v.mp4
```

> **Note**: The 5B model is fast — 40 steps run quickly and are recommended for best quality.

> **Frame count**: `--num-frames` must satisfy `4n+1` for all models (e.g. 5, 9, 13, 21, 41, 81, 101 …).

> **Resolution**: Always use the model's native resolution. While generation will succeed at other sizes, mismatched resolutions or aspect ratios are likely to produce visual artifacts. Preferred resolutions are:
> - **480P** — 832×480 (landscape) or 480×832 (portrait) — for Wan2.1 1.3B
> - **720P** — 1280×704 (landscape) or 704×1280 (portrait) — for Wan2.1 14B, Wan2.2 T2V/I2V/TI2V

#### Config Files

WAN generation accepts JSON, YAML, and YML run configs with keys matching the CLI option names, using underscores instead of dashes. Explicit CLI flags override config values.

```yaml
# wan-run.yaml
model_dir: ./Wan2.2-T2V-A14B-MLX
prompt:
  - "Two astronauts playing chess on the surface of the moon, dramatic lighting"
  - "A rover crossing a moonlit crater, dramatic lighting"
negative_prompt: "low quality, blurry, distorted"
width: 1280
height: 704
num_frames: 81
steps: 40
guide_scale: "3.0,4.0"
refiner_start: 0.125
sigma_schedule: official
legacy_vae_decode: false
positive_conditioning_npz: null
negative_conditioning_npz: null
dump_text_conditioning_npz: null
dump_final_latents_npz: null
initial_latents_npz: null
seed: 42
output_path: output/wan22_t2v
```

For Wan2.2 dual-model runs, `refiner_start` controls when generation switches
from the high-noise model to the low-noise model. A fraction such as `0.125`
uses `ceil(steps * 0.125)` high-noise steps, so `steps: 8` starts the low-noise
model at step 2. Integer values are one-based low-noise start steps; `1` means
low-noise from the first step and `steps + 1` means all high-noise.

```bash
python -m mlx_video.wan_2.generate --config wan-run.yaml
```

JSON uses the same shape. `prompt` may be a string or a list of strings; list values generate one output per prompt. Bare config names such as `alt.yaml` are resolved from the repo `configs/` directory; explicit paths such as `./alt.yaml`, `configs/alt.yaml`, `../alt.yaml`, and absolute paths are used as written. When a config has a sibling `_default.yaml`, missing fields are filled from that file. Values in the requested config override `_default.yaml`, and explicit CLI flags override both.

```json
{
  "model_dir": "./Wan2.2-T2V-A14B-MLX",
  "prompt": [
    "A slow tracking shot through a neon market at night",
    "A rain-soaked alley filled with neon signs"
  ],
  "width": 1280,
  "height": 704,
  "num_frames": 81,
  "steps": 40,
  "guide_scale": "3.0,4.0",
  "seed": 7,
  "output_path": "market.mp4"
}
```

Repeat `--config` to generate multiple videos sequentially:

```bash
python -m mlx_video.wan_2.generate \
    --config first.yaml \
    --config second.json \
    --config third.yml
```

Supported config keys are: `model_dir`, `prompt`, `image`, `negative_prompt`, `no_negative_prompt`, `width`, `height`, `num_frames`, `steps`, `guide_scale`, `shift`, `refiner_start`, `seed`, `output_path`, `fps`, `output_last_frame`, `scheduler`, `sigma_schedule`, `noise_source`, `torch_python`, `positive_conditioning_npz`, `negative_conditioning_npz`, `dump_text_conditioning_npz`, `dump_final_latents_npz`, `initial_latents_npz`, `lora`, `lora_high`, `lora_low`, `tiling`, `legacy_vae_decode`, `no_compile`, `trim_first_frames`, `debug_latents`, `iterations`, `iteration_seed`, and `output_template`.

#### Bridge Diagnostics

The NPZ bridge options are diagnostic tools for isolating where parity diverges:

- `--initial-latents-npz` loads exact starting latents/noise instead of using RNG.
- `--positive-conditioning-npz` and `--negative-conditioning-npz` load raw text conditioning and bypass MLX T5 encoding.
- `--dump-text-conditioning-npz` saves MLX raw T5 conditioning.
- `--dump-final-latents-npz` saves final denoised latents before VAE decode.

#### Generation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | — | JSON/YAML run config; repeat for sequential batch generation |
| `--model-dir` | (required) | Path to converted MLX model directory |
| `--prompt` | (required) | Text prompt; repeat to generate one output per prompt |
| `--image` | — | Input image path (I2V and TI2V modes) |
| `--negative-prompt` | config default | Negative guidance prompt |
| `--width` | `1280` | Output width in pixels |
| `--height` | `704` | Output height in pixels |
| `--num-frames` | `81` | Number of frames (must be `4n+1`) |
| `--steps` | config default | Diffusion steps |
| `--guide-scale` | config default | Guidance scale; use `"high,low"` pair for Wan2.2 dual models |
| `--shift` | config default | Noise schedule shift |
| `--refiner-start` | model default | Dual-model low-noise start: fraction in `(0, 1)` or one-based step `1..steps+1` |
| `--seed` | `-1` (random) | Random seed for reproducibility |
| `--output-path` | `output.mp4` | Output video file path |
| `--fps` | config default | Output video frames per second |
| `--output-last-frame` / `--no-output-last-frame` | auto | Save sibling PNG of the final frame; enabled by default for one-frame runs |
| `--iterations` | `1` | Run multiple generations; when greater than 1, `--output-path` is treated as an output directory |
| `--iteration-seed` | `increment` | Seed strategy for iterations: `same`, `increment`, or `random` |
| `--output-template` | model default | Relative output filename template used under `--output-path` |
| `--scheduler` | `unipc` | Solver: `euler`, `dpm++`, or `unipc` |
| `--sigma-schedule` | `official` | Sigma schedule: `official` or `comfy-simple` |
| `--noise-source` | `mlx` | Initial latent noise source: `mlx` or `torch`; use `torch` for PyTorch parity checks |
| `--torch-python` | active Python | Python executable used to generate Torch noise for `--noise-source torch` |
| `--initial-latents-npz` | — | Load exact initial latents/noise from NPZ instead of RNG |
| `--positive-conditioning-npz` | — | Load raw positive text conditioning from NPZ and bypass MLX T5 |
| `--negative-conditioning-npz` | — | Load raw negative text conditioning when CFG is enabled |
| `--dump-text-conditioning-npz` | — | Save MLX raw text conditioning to NPZ |
| `--dump-final-latents-npz` | — | Save final denoised latents to NPZ before VAE decode |
| `--trim-first-frames` | `0` | Drop N leading frames (fixes first-frame artifacts on 14B models) |
| `--tiling` | `auto` | VAE tiling: `auto`, `none`, `spatial`, `temporal` |
| `--legacy-vae-decode` | off | Use the old Wan2.1/14B single-frame VAE temporal upsample path instead of reference decoding |

#### Output Filename Templates

`--output-template` uses Python `str.format` fields and renders a relative path under `--output-path`. If the rendered filename has no extension, `.mp4` is appended. Nested directories are created automatically.

```bash
python -m mlx_video.wan_2.generate \
    --model-dir ./Wan2.2-T2V-A14B-MLX \
    --prompt "A silver train crossing a desert bridge" \
    --iterations 3 \
    --output-path output/runs \
    --output-template "{mode}/seed{seed}/iter{iteration1:03d}-{size}"
```

The default Wan template is:

```text
wan-{mode}-seed{seed}-s{steps}-sh{shift}-{width}x{height}.mp4
```

| Field | Example | Description |
|-------|---------|-------------|
| `{model}` | `wan` | Model family |
| `{mode}` | `t2v` | Generation mode: `t2v` or `i2v` |
| `{seed}` | `4000000033` | Concrete seed for the iteration |
| `{steps}` | `8` | Diffusion steps, compact-formatted |
| `{shift}` | `5` | Noise shift, compact-formatted |
| `{width}` | `1280` | Output width |
| `{height}` | `704` | Output height |
| `{frames}` | `81` | Output frame count |
| `{fps}` | `16` | Output frames per second |
| `{iteration}` | `0` | Zero-based iteration index |
| `{iteration1}` | `1` | One-based iteration number; supports format specs like `{iteration1:03d}` |
| `{size}` | `1280x704` | Width and height joined as `WIDTHxHEIGHT` |
| `{mode_part}` | `t2v-` | Mode plus trailing dash, empty when unavailable |
| `{steps_part}` | `s8` | Steps with `s` prefix |
| `{shift_part}` | `sh5` | Shift with `sh` prefix, empty when unavailable |

### Quantization (Reduced Memory)

Quantize the transformer weights to reduce memory usage by ~3.4×. Quantization is supported for all model variants and is especially important for running 14B models on devices with limited unified memory:

```bash
# Convert with 4-bit quantization (works for any variant)
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.1-T2V-1.3B \
    --output-dir ./Wan2.1-T2V-1.3B-MLX-Q4 \
    --quantize --bits 4 --group-size 64

python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.1-T2V-14B \
    --output-dir ./Wan2.1-T2V-14B-MLX-Q4 \
    --quantize --bits 4 --group-size 64

python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-T2V-A14B \
    --output-dir ./Wan2.2-T2V-A14B-MLX-Q4 \
    --quantize --bits 4 --group-size 64

python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-I2V-A14B \
    --output-dir ./Wan2.2-I2V-A14B-MLX-Q4 \
    --quantize --bits 4 --group-size 64

python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-TI2V-5B \
    --output-dir ./Wan2.2-TI2V-5B-MLX-Q4 \
    --quantize --bits 4 --group-size 64
```

You can also quantize an already-converted MLX model without re-converting from PyTorch:

```bash
python -m mlx_video.wan2.convert \
    --checkpoint-dir ./Wan2.2-T2V-A14B-MLX \
    --output-dir ./Wan2.2-T2V-A14B-MLX-Q4 \
    --quantize-only --bits 4
```

Quantized models are used exactly the same way — the quantization is auto-detected from `config.json`:

```bash
python -m mlx_video.wan2.generate \
    --model-dir ./Wan2.2-T2V-A14B-MLX-Q4 \
    --prompt "A cat playing piano"
```

**What gets quantized**: Self-attention (Q/K/V/O), cross-attention (Q/K/V/O), and FFN (fc1/fc2) — 10 layers × N blocks = ~95% of model weights. Embeddings, norms, and the output head remain in bfloat16 for precision.

| Model | BF16 Size | 4-bit Size | Notes |
|-------|-----------|------------|-------|
| 1.3B | 2.7 GB | 799 MB | ~3.4x smaller |
| 14B | ~28 GB | ~8 GB | Enables running on 16GB devices |

> **Note**: On Apple Silicon, the 1.3B model fits comfortably in unified memory at bf16. Quantization reduces memory but may not speed up inference for small models. For the 14B model, quantization is essential to fit in memory and will also improve speed.

### Wan Model Specifications

**Transformer (14B)**
- 40 layers, 40 attention heads, dim 5120, head dim 128
- 3-way factorized RoPE (temporal + spatial)
- 14.29B parameters

**Transformer (1.3B, Wan2.1 only)**
- 30 layers, 12 attention heads, dim 1536, head dim 128
- Same architecture, smaller scale

**Text Encoder** — UMT5-XXL (5.68B parameters)
- 24 layers, 64 heads, dim 4096, vocab 256K

**VAE** — 3D causal convolution decoder (72.6M parameters)
- Latent channels: 16
- Compression: 4× temporal, 8× spatial
- Wan2.1 decoding uses chunked temporal cache handling for multi-frame latents
  so tiled and non-tiled decode preserve the same frame contract.

---

## LoRA Support

LoRA's can be used with the `--lora-high` and `--lora-low` command line switches.
Use `--no-lora`, `--no-lora-high`, or `--no-lora-low` to clear LoRAs inherited
from a config file. In config files, use `lora: []`, `lora_high: []`, or
`lora_low: []` to clear inherited defaults.

For example, to use a distilled Wan2.2-Lightning LoRA, use the following command. Lightning speeds up generation by using only 4 steps and a CFG scale of 1.

```bash
python -m mlx_video.wan2.generate \
    --model-dir /Volumes/SSD/Wan-AI/Wan2.2-T2V-A14B-MLX \
    --width 480 \
    --height 704 \
    --num-frames 41 \
    --prompt "Two dogs of the poodle breed sitting on a beach wearing sunglasses, nodding with their heads, close up, cinematic, sunset" \
    --steps 4 \
    --guide-scale 1 \
    --trim-first-frames 1 \
    --seed 2391784614 \
    --lora-high /path/to/Wan2.2-Lightning/high_noise_model.safetensors 1 \
    --lora-low /path/to/Wan2.2-Lightning/low_noise_model.safetensors 1
 ```

## Enjoy

![Poodles](../../../examples/poodles-wan.gif)

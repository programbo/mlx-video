"""Wan model loading utilities."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def load_wan_model(
    model_path: Path,
    config,
    quantization: dict | None = None,
    loras: list | None = None,
):
    """Load and initialize WanModel, with optional quantization and LoRA support.

    Args:
        model_path: Path to model safetensors file
        config: WanModelConfig
        quantization: Optional dict with 'bits' and 'group_size' keys.
                      If provided, creates QuantizedLinear stubs before loading.
        loras: Optional list of (lora_path, strength) tuples to apply.
    """
    from mlx_video.models.wan_2.wan_2 import WanModel

    model = WanModel(config)

    if quantization:
        from mlx_video.models.wan_2.convert import _quantize_predicate

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, m: _quantize_predicate(path, m),
        )

    weights = mx.load(str(model_path))

    # Apply LoRAs: dequantize+merge for quantized models, weight merge for bf16
    if loras:
        if quantization:
            # Dequantize LoRA-targeted layers, merge delta, replace with bf16 Linear.
            # Non-LoRA layers stay 4-bit. Zero per-step overhead.
            from mlx_video.models.wan_2.convert import _load_lora_configs
            from mlx_video.lora import apply_loras_to_model

            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            module_to_loras = _load_lora_configs(loras)
            apply_loras_to_model(model, module_to_loras)
            mx.eval(model.parameters())
            return model
        else:
            # Weight merging: fold LoRA into bf16 weights before loading
            from mlx_video.models.wan_2.convert import load_and_apply_loras

            weights = load_and_apply_loras(dict(weights), loras)

    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def load_t5_encoder(model_path: Path, config):
    """Load T5 text encoder.

    Weights are upcast to float32 for maximum precision — the T5 encoder
    only runs once per generation, so performance impact is negligible.
    This matches the official which computes softmax in float32 explicitly.
    """
    from mlx_video.models.wan_2.text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )
    weights = mx.load(str(model_path))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())
    return encoder


def load_t5_encoder_fp8_scaled(model_path: Path, config):
    """Load a Comfy-style scaled-fp8 UMT5 encoder into the MLX T5 module.

    The supported file layout uses HuggingFace/Comfy key names such as
    ``encoder.block.0.layer.0.SelfAttention.q.weight`` plus scalar
    ``scale_weight`` tensors for float8 matrices.
    """
    from safetensors import safe_open

    from mlx_video.models.wan_2.text_encoder import T5Encoder

    encoder = T5Encoder(
        vocab_size=config.t5_vocab_size,
        dim=config.t5_dim,
        dim_attn=config.t5_dim_attn,
        dim_ffn=config.t5_dim_ffn,
        num_heads=config.t5_num_heads,
        num_layers=config.t5_num_layers,
        num_buckets=config.t5_num_buckets,
        shared_pos=False,
    )

    def _scaled_tensor(handle, key: str) -> mx.array:
        tensor = handle.get_tensor(key)
        if "float8" in str(tensor.dtype):
            scale_key = key.removesuffix(".weight") + ".scale_weight"
            if scale_key not in handle.keys():
                raise ValueError(
                    f"Scaled-fp8 text encoder is missing scale tensor: {scale_key}"
                )
            tensor = tensor.float() * handle.get_tensor(scale_key).float()
        else:
            tensor = tensor.float()
        return mx.array(tensor.cpu().numpy()).astype(mx.float32)

    weights = {
        "token_embedding.weight": None,
        "norm.weight": None,
    }
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        weights["token_embedding.weight"] = _scaled_tensor(handle, "shared.weight")
        weights["norm.weight"] = _scaled_tensor(
            handle, "encoder.final_layer_norm.weight"
        )
        for layer in range(config.t5_num_layers):
            prefix = f"encoder.block.{layer}"
            target = f"blocks.{layer}"
            weights[f"{target}.norm1.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.layer_norm.weight"
            )
            weights[f"{target}.attn.q.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.SelfAttention.q.weight"
            )
            weights[f"{target}.attn.k.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.SelfAttention.k.weight"
            )
            weights[f"{target}.attn.v.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.SelfAttention.v.weight"
            )
            weights[f"{target}.attn.o.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.SelfAttention.o.weight"
            )
            weights[f"{target}.pos_embedding.embedding.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
            )
            weights[f"{target}.norm2.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.1.layer_norm.weight"
            )
            weights[f"{target}.ffn.gate_proj.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.1.DenseReluDense.wi_0.weight"
            )
            weights[f"{target}.ffn.fc1.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.1.DenseReluDense.wi_1.weight"
            )
            weights[f"{target}.ffn.fc2.weight"] = _scaled_tensor(
                handle, f"{prefix}.layer.1.DenseReluDense.wo.weight"
            )

    encoder.load_weights(list(weights.items()))
    mx.eval(encoder.parameters())
    return encoder


def load_vae_decoder(model_path: Path, config=None):
    """Load VAE decoder (skips encoder weights with strict=False).

    For Wan2.2 (vae_z_dim=48), uses Wan22VAEDecoder.
    For Wan2.1 (vae_z_dim=16), uses WanVAE.
    """
    is_wan22 = config is not None and config.vae_z_dim == 48

    if is_wan22:
        from mlx_video.models.wan_2.vae22 import Wan22VAEDecoder

        vae = Wan22VAEDecoder(z_dim=48)
    else:
        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16)

    weights = mx.load(str(model_path))
    # Upcast VAE weights to float32 for quality — official Wan2.2 runs VAE in float32
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def load_vae_encoder(model_path: Path, config=None):
    """Load VAE encoder for I2V image encoding.

    For Wan2.2 TI2V (vae_z_dim=48), uses Wan22VAEEncoder.
    For Wan2.1/I2V-14B (vae_z_dim=16), uses WanVAE with encoder=True.
    """
    if config is not None and config.vae_z_dim == 16:
        from mlx_video.models.wan_2.vae import WanVAE

        vae = WanVAE(z_dim=16, encoder=True)
    else:
        from mlx_video.models.wan_2.vae22 import Wan22VAEEncoder

        vae = Wan22VAEEncoder(z_dim=config.vae_z_dim if config else 48)

    weights = mx.load(str(model_path))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    vae.load_weights(list(weights.items()), strict=False)
    mx.eval(vae.parameters())
    return vae


def _clean_text(text: str) -> str:
    """Clean text matching official Wan2.2 tokenizer preprocessing.

    Applies ftfy.fix_text (fixes mojibake, normalizes fullwidth chars),
    double HTML unescape, and whitespace normalization. Critical for
    correct tokenization of the Chinese negative prompt.
    """
    import html
    import re

    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(
    encoder,
    tokenizer,
    prompt: str,
    text_len: int = 512,
) -> mx.array:
    """Encode text prompt using T5 encoder.

    Args:
        encoder: T5Encoder model
        tokenizer: HuggingFace tokenizer
        prompt: Text prompt
        text_len: Maximum text length

    Returns:
        Text embeddings [L, dim]
    """
    prompt = _clean_text(prompt)
    tokens = tokenizer(
        prompt,
        max_length=text_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    ids = mx.array(tokens["input_ids"])
    mask = mx.array(tokens["attention_mask"])

    embeddings = encoder(ids, mask=mask)

    # Return only non-padding tokens
    seq_len = int(mask.sum().item())
    return embeddings[0, :seq_len]

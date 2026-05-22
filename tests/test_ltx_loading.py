import json


def test_ltx_transformer_uses_indexed_safetensor_shards(tmp_path):
    from mlx_video.models.ltx_2.ltx_2 import _indexed_safetensor_files

    expected = tmp_path / "model-00000-of-00002.safetensors"
    other = tmp_path / "model-00001-of-00002.safetensors"
    extra = tmp_path / "model-00001-of-00008.safetensors"
    expected.touch()
    other.touch()
    extra.touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "blocks.0.weight": expected.name,
                    "blocks.1.weight": other.name,
                }
            }
        )
    )

    assert _indexed_safetensor_files(tmp_path) == [expected, other]


def test_ltx_text_encoder_prefers_sibling_tokenizer_for_nested_encoder(tmp_path):
    from mlx_video.models.ltx_2.text_encoder import _resolve_text_encoder_paths

    model_dir = tmp_path / "ltx-model"
    encoder_root = tmp_path / "gemma"
    (model_dir / "tokenizer").mkdir(parents=True)
    (encoder_root / "text_encoder").mkdir(parents=True)
    (encoder_root / "tokenizer").mkdir()

    text_encoder_path, tokenizer_candidates = _resolve_text_encoder_paths(
        model_dir, encoder_root
    )

    assert text_encoder_path == str(encoder_root / "text_encoder")
    assert tokenizer_candidates[0] == encoder_root / "tokenizer"
    assert model_dir / "tokenizer" in tokenizer_candidates

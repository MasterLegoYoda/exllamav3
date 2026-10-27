"""
LFM2.5 config parsing and image preprocessing details that the LFM2-VL path depends on:

- The checkpoint families spell weight tying as `tie_embedding`, or omit the flag entirely and ship
  no `lm_head` tensor. With no head tensor there is nothing else to load and the reference ties the
  head to the input embedding, so the config must preserve that rather than leave the flag false.
- LFM2.5-VL's image processor uses bilinear resampling (PIL enum 2), while the earlier LFM2-VL
  release used bicubic (3). The preprocessing helpers must honor the config instead of assuming
  bicubic.

Synthetic configs, no weights.
"""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file

from exllamav3 import Config

pytestmark = pytest.mark.nogpu


LFM2_TEXT = {
    "architectures": ["Lfm2ForCausalLM"],
    "model_type": "lfm2",
    "hidden_size": 128,
    "head_dim": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 256,
    "hidden_act": "silu",
    "norm_eps": 1e-5,
    "conv_L_cache": 3,
    "vocab_size": 1024,
    "layer_types": ["conv", "conv", "full_attention", "conv"],
    "rope_parameters": {"rope_theta": 1000000.0, "rope_type": "default"},
}

LFM2_VL = {
    "architectures": ["Lfm2VlForConditionalGeneration"],
    "model_type": "lfm2_vl",
    "text_config": copy.deepcopy(LFM2_TEXT),
    "vision_config": {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "patch_size": 8,
        "num_patches": 16,
        "layer_norm_eps": 1e-6,
        "hidden_act": "gelu_pytorch_tanh",
        "num_channels": 3,
    },
    "downsample_factor": 2,
    "projector_hidden_size": 64,
    "projector_hidden_act": "gelu",
    "projector_use_layernorm": False,
    "use_image_special_tokens": True,
    "image_token_id": 100,
}

LFM2_VL_PROCESSOR = {
    "image_processor": {
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255,
        "encoder_patch_size": 8,
        "downsample_factor": 2,
        "min_image_tokens": 16,
        "max_image_tokens": 64,
        "min_tiles": 1,
        "max_tiles": 1,
        "tile_size": 64,
        "use_thumbnail": False,
        "do_image_splitting": False,
        "max_pixels_tolerance": 2.0,
        "resample": 2,
    },
}


def write_config(tmp_path, config, processor = None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if processor is not None:
        (tmp_path / "processor_config.json").write_text(json.dumps(processor))
    return str(tmp_path)


def load_config(tmp_path, config, processor = None):
    return Config.from_directory(write_config(tmp_path, config, processor))


def test_tie_embedding_alias(tmp_path):
    # Converted LFM2.5 configs can carry both keys, with tie_embedding being the authoritative one. They can
    # also carry a quantized clone of the tied head, which must not override the explicit tie flag.
    save_file({"lm_head.trellis": torch.zeros(1, 1, dtype = torch.uint8)}, tmp_path / "model.safetensors")
    config = copy.deepcopy(LFM2_TEXT)
    config["tie_embedding"] = True
    config["tie_word_embeddings"] = False
    assert load_config(tmp_path, config).tie_word_embeddings


def test_missing_lm_head_implies_tied(tmp_path):
    # LFM2.5-VL checkpoints omit both tie flags and ship no lm_head; the embedding is the head
    assert load_config(tmp_path, LFM2_VL, LFM2_VL_PROCESSOR).tie_word_embeddings
    assert load_config(tmp_path, copy.deepcopy(LFM2_TEXT)).tie_word_embeddings


def test_lm_head_tensor_keeps_untied_config_untied(tmp_path):
    save_file({"lm_head.weight": torch.zeros(1, 1, dtype = torch.float16)}, tmp_path / "model.safetensors")
    # No explicit tie flag, but a real head exists: do not silently redirect it to the embedding
    assert not load_config(tmp_path, copy.deepcopy(LFM2_TEXT)).tie_word_embeddings


def test_quantized_lm_head_keeps_untied_config_untied(tmp_path):
    save_file({"lm_head.trellis": torch.zeros(1, 1, dtype = torch.uint8)}, tmp_path / "model.safetensors")
    assert not load_config(tmp_path, copy.deepcopy(LFM2_TEXT)).tie_word_embeddings


def test_lfm25_vl_bilinear_resample(tmp_path):
    config = load_config(tmp_path, LFM2_VL, LFM2_VL_PROCESSOR)
    assert config.vision_pp.resample == 2


def test_lfm2_vl_bicubic_resample(tmp_path):
    processor = copy.deepcopy(LFM2_VL_PROCESSOR)
    processor["image_processor"]["resample"] = 3
    config = load_config(tmp_path, LFM2_VL, processor)
    assert config.vision_pp.resample == 3


def test_resize_honors_resample_mode():
    from exllamav3.architecture.mm_processing.lfm2_vl import lfm2_vl_resize

    g = torch.Generator().manual_seed(5)
    pixels = torch.randint(0, 256, (3, 16, 16), generator = g, dtype = torch.uint8)
    bilinear = lfm2_vl_resize(pixels, (9, 9), 2)
    bicubic = lfm2_vl_resize(pixels, (9, 9), 3)
    assert bilinear.shape == bicubic.shape == (3, 9, 9)
    assert not torch.equal(bilinear, bicubic)


def test_unsupported_resample_rejected(tmp_path):
    processor = copy.deepcopy(LFM2_VL_PROCESSOR)
    processor["image_processor"]["resample"] = 1
    with pytest.raises(AssertionError, match = "Unsupported resampling mode"):
        load_config(tmp_path, LFM2_VL, processor)

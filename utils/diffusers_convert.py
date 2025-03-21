import re
import torch
import logging

# conversion code from https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_stable_diffusion.py

# ================#
# VAE Conversion #
# ================#

vae_conversion_map = [
    # (stable-diffusion, HF Diffusers)
    ("nin_shortcut", "conv_shortcut"),
    ("norm_out", "conv_norm_out"),
    ("mid.attn_1.", "mid_block.attentions.0."),
]

for i in range(4):
    # down_blocks have two resnets
    for j in range(2):
        hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
        sd_down_prefix = f"encoder.down.{i}.block.{j}."
        vae_conversion_map.append((sd_down_prefix, hf_down_prefix))

    if i < 3:
        hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
        sd_downsample_prefix = f"down.{i}.downsample."
        vae_conversion_map.append((sd_downsample_prefix, hf_downsample_prefix))

        hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
        sd_upsample_prefix = f"up.{3 - i}.upsample."
        vae_conversion_map.append((sd_upsample_prefix, hf_upsample_prefix))

    # up_blocks have three resnets
    # also, up blocks in hf are numbered in reverse from sd
    for j in range(3):
        hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
        sd_up_prefix = f"decoder.up.{3 - i}.block.{j}."
        vae_conversion_map.append((sd_up_prefix, hf_up_prefix))

# this part accounts for mid blocks in both the encoder and the decoder
for i in range(2):
    hf_mid_res_prefix = f"mid_block.resnets.{i}."
    sd_mid_res_prefix = f"mid.block_{i + 1}."
    vae_conversion_map.append((sd_mid_res_prefix, hf_mid_res_prefix))

vae_conversion_map_attn = [
    # (stable-diffusion, HF Diffusers)
    ("norm.", "group_norm."),
    ("q.", "query."),
    ("k.", "key."),
    ("v.", "value."),
    ("q.", "to_q."),
    ("k.", "to_k."),
    ("v.", "to_v."),
    ("proj_out.", "to_out.0."),
    ("proj_out.", "proj_attn."),
]


def reshape_weight_for_sd(w):
    # convert SD conv2d weights to HF linear weights
    w.view(w.shape[0], w.shape[1])


def convert_vae_state_dict_to_hf(vae_state_dict):
    mapping = {k: k for k in vae_state_dict.keys()}
    conv3d = False
    for k, v in mapping.items():
        for sd_part, hf_part in vae_conversion_map:
            v = v.replace(sd_part, hf_part)
        # if v.endswith(".conv.weight"):
        #     if not conv3d and vae_state_dict[k].ndim == 5:
        #         conv3d = True
        mapping[k] = v
    for k, v in mapping.items():
        if "attentions" in k:
            for sd_part, hf_part in vae_conversion_map_attn:
                v = v.replace(sd_part, hf_part)
            mapping[k] = v
    new_state_dict = {v: vae_state_dict[k] for k, v in mapping.items()}
    weights_to_convert = ["to_q", "to_k", "to_v", "to_out.0", "proj_attn", "value", "key", "query"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid_block.attentions.0.{weight_name}.weight" in k:
                logging.debug(f"Reshaping {k} for HF format")
                new_state_dict[k] = reshape_weight_for_sd(v)
    return new_state_dict

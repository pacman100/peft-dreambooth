import argparse
import os
from typing import Dict

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel

from peft import PeftModel, get_peft_model_state_dict

LORA_ADAPTER_NAME = "default"


def combine_unet_and_text_encoder(
    sd_checkpoint, peft_lora_path, dump_path, adapter_name=LORA_ADAPTER_NAME, half=True, sd_checkpoint_revision=None
):
    # Store PEFT Checkpoint as a single `safetensors` file
    metadata = {}
    state_dict = {}
    dtype = torch.float16 if half else torch.float32
    sd_checkpoint_revision = sd_checkpoint_revision if sd_checkpoint_revision != "" else None

    # Load Text Encoder LoRA model
    text_encoder_peft_lora_path = os.path.join(peft_lora_path, "text_encoder")
    if os.path.exists(text_encoder_peft_lora_path):
        text_encoder = CLIPTextModel.from_pretrained(
            sd_checkpoint, subfolder="text_encoder", revision=sd_checkpoint_revision
        )
        text_encoder = PeftModel.from_pretrained(text_encoder, text_encoder_peft_lora_path, adapter_name=adapter_name)
        text_encoder_state_dict = get_peft_model_state_dict(text_encoder, adapter_name=adapter_name)
        text_encoder_state_dict = {k: v.to(dtype) for k, v in text_encoder_state_dict.items()}
        state_dict.update(text_encoder_state_dict)
        target_modules_as_text = ",".join(getattr(text_encoder.peft_config[adapter_name], "target_modules"))
        metadata["text_encoder_target_modules"] = target_modules_as_text

    # Load UNet LoRA model
    unet_peft_lora_path = os.path.join(peft_lora_path, "unet")
    if os.path.exists(unet_peft_lora_path):
        unet = UNet2DConditionModel.from_pretrained(sd_checkpoint, subfolder="unet", revision=sd_checkpoint_revision)
        unet = PeftModel.from_pretrained(unet, unet_peft_lora_path, adapter_name=adapter_name)
        unet_state_dict = get_peft_model_state_dict(unet, adapter_name=adapter_name)
        unet_state_dict = {k: v.to(dtype) for k, v in unet_state_dict.items()}
        state_dict.update(unet_state_dict)
        target_modules_as_text = ",".join(getattr(unet.peft_config[adapter_name], "target_modules"))
        metadata["unet_target_modules"] = target_modules_as_text
        for param in ["r", "lora_alpha", "lora_dropout"]:
            metadata[param] = str(getattr(unet.peft_config[adapter_name], param))

    print(f"{metadata=}")
    print(f"saving the converted ckpt to {dump_path=}")
    # Save state dict
    save_file(state_dict, dump_path, metadata=metadata)
    print(f"Saved the converted ckpt to {dump_path=}! 🤗🚀✨")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sd_checkpoint",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--sd_checkpoint_revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument("--peft_lora_path", default=None, type=str, required=True, help="Path to peft trained LoRA")

    parser.add_argument("--adapter_name", default=LORA_ADAPTER_NAME, type=str, required=False, help="Adapter Name")

    parser.add_argument(
        "--dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output safetensors file for use with webui.",
    )

    parser.add_argument("--half", action="store_true", help="Save weights in half precision.")
    args = parser.parse_args()
    combine_unet_and_text_encoder(
        args.sd_checkpoint,
        args.peft_lora_path,
        args.dump_path,
        adapter_name=args.adapter_name,
        half=args.half,
        sd_checkpoint_revision=args.sd_checkpoint_revision,
    )

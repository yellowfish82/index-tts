#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import gc
import argparse
from indextts.infer_v2 import IndexTTS2

# 全局模型对象
_tts_model = None


def init_model(cfg_path="checkpoints/config.yaml",
               model_dir="checkpoints",
               use_fp16=True,
               use_cuda_kernel=False,
               use_deepspeed=False):
    """
    初始化 IndexTTS2 模型（仅 GPU）
    """
    global _tts_model
    if _tts_model is None:
        gc.collect()
        torch.cuda.empty_cache()

        print("[INFO] 正在加载 IndexTTS2 模型到 GPU ...")
        _tts_model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed
        )
    return _tts_model


def tts_infer_safe(text, spk_audio_prompt, output_path):
    """
    使用已初始化模型进行语音合成
    """
    global _tts_model
    if _tts_model is None:
        raise RuntimeError("Model is not initialized. Call init_model() first.")

    gc.collect()
    torch.cuda.empty_cache()

    try:
        _tts_model.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=output_path,
            verbose=True
        )
    except torch.cuda.OutOfMemoryError:
        print("[ERROR] CUDA OOM! 请尝试缩短输入文本或更换更大显存的显卡。")
        raise


def main():
    parser = argparse.ArgumentParser(description="IndexTTS2 GPU inference script (safe version)")
    parser.add_argument("text", type=str, help="Text to synthesize")
    parser.add_argument("spk_audio_prompt", type=str, help="Path to reference speaker audio")
    parser.add_argument("output_path", type=str, help="Path to save generated audio")
    args = parser.parse_args()

    gc.collect()
    torch.cuda.empty_cache()

    # 初始化模型（只初始化一次）
    init_model()

    # 推理
    tts_infer_safe(args.text, args.spk_audio_prompt, args.output_path)


if __name__ == "__main__":
    main()


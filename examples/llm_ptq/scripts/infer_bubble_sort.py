#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 fakequant 推理脚本: 让 Qwen3.5-0.8B 写一个冒泡排序.

流程:
1) 加载 BF16 模型 (复用 hf_ptq 的 get_model, 能正确识别 VLM 架构)
2) 用少量样本做 NVFP4 全量量化校准 (和 quantize_full_nvfp4.py 同一条 quant_cfg)
3) 直接在 fakequant 状态下 model.generate(), 流式打印

为什么走 fakequant 而不是 load 已导出的 NVFP4 checkpoint:
- 导出的 HF checkpoint 是 packed FP4 (uint8 存储), transformers 没有 NVFP4
  的反量化支持, AutoModelForCausalLM 重载时 shape mismatch
- 在本机 (Ampere) 上也没有 NVFP4 tensor core, 只能走 fakequant = BF16 GEMM
  + 每次 matmul 前后 quant/dequant. 这是本项目 examples/vllm_serve/ 同款思路
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from pathlib import Path

# modelopt 的 dynamic-block quantize 依赖现编的 CUDA 扩展, 要 CUDA 12+ 的
# libcudacxx (cuda/std/bit). 远程机默认的 /usr/bin/nvcc 是 CUDA 11.5, 头文件
# 不全; 优先用 /usr/local/cuda-12.6 的 toolkit. 用户可通过 CUDA_HOME 覆盖.
if "CUDA_HOME" not in os.environ:
    for _cuda in ("/usr/local/cuda-12.6", "/usr/local/cuda-12.8", "/usr/local/cuda"):
        if os.path.isfile(os.path.join(_cuda, "bin", "nvcc")):
            os.environ["CUDA_HOME"] = _cuda
            os.environ["PATH"] = f"{_cuda}/bin:" + os.environ.get("PATH", "")
            break

import numpy as np
import torch
from transformers import TextStreamer

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

_PTQ_DIR = Path(__file__).resolve().parent.parent
if str(_PTQ_DIR) not in sys.path:
    sys.path.insert(0, str(_PTQ_DIR))
from example_utils import get_model, get_tokenizer  # noqa: E402

RAND_SEED = 1234
DEFAULT_PROMPT = "请用 Python 写一个冒泡排序算法, 包含完整函数、一个测试用例和注释."


def build_full_nvfp4_cfg() -> dict:
    """ALL 配置: 所有 Linear/Conv 的 weight + input 全走 NVFP4, 不附加任何默认排除项.

    等价于 quantize_full_nvfp4.py 的 quant_cfg (对应 Qwen3.5-0.8B-NVFP4-ALL-T1).
    """
    nvfp4 = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    }
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*weight_quantizer", "cfg": copy.deepcopy(nvfp4)},
            {"quantizer_name": "*input_quantizer", "cfg": copy.deepcopy(nvfp4)},
        ],
        "algorithm": "max",
    }


def build_default_nvfp4_cfg() -> dict:
    """DEFAULT 配置: modelopt 自带 NVFP4_DEFAULT_CFG, 排除 lm_head / linear_attn.conv1d
    / mlp.gate / router / output_layer 等 (对应 Qwen3.5-0.8B-NVFP4-T1 语义).
    """
    return copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/disk1/guohaoran/models/Qwen3.5-0.8B",
                        help="HF 源模型目录 (BF16)")
    parser.add_argument("--dataset", default="/data/disk1/guohaoran/calib_text_128.jsonl",
                        help="校准数据集 (本地 .jsonl 或已注册名)")
    parser.add_argument("--calib_size", type=int, default=16,
                        help="校准样本数, 推理场景量少即可")
    parser.add_argument("--calib_seq", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--cfg", choices=["all", "default"], default="all",
                        help="NVFP4 量化范围: all=所有 Linear/Conv 的 weight+input "
                             "(对齐 Qwen3.5-0.8B-NVFP4-ALL-T1); default=modelopt "
                             "自带 NVFP4_DEFAULT_CFG, 跳过 lm_head/linear_attn.conv1d 等")
    parser.add_argument("--skip_quant", action="store_true",
                        help="跳过量化, 直接用 BF16 模型推理 (用于对比)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise OSError("infer_bubble_sort: CUDA GPU required")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    mto.enable_huggingface_checkpointing()

    # ------------------------------------------------------------------
    # 1) 加载模型
    # ------------------------------------------------------------------
    print(f"infer_bubble_sort: loading {args.model}")
    model = get_model(args.model, device="cuda", trust_remote_code=args.trust_remote_code)
    model.eval()
    tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2) NVFP4 fakequant 校准 (可选)
    # ------------------------------------------------------------------
    if not args.skip_quant:
        quant_cfg = build_full_nvfp4_cfg() if args.cfg == "all" else build_default_nvfp4_cfg()
        print(f"infer_bubble_sort: calibrating NVFP4 ({args.cfg}) with {args.calib_size} samples")
        calib_loader = get_dataset_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_samples=args.calib_size,
            max_sample_length=args.calib_seq,
            device=model.device,
        )
        mtq.quantize(
            model,
            quant_cfg,
            forward_loop=create_forward_loop(dataloader=calib_loader),
        )
        print(f"infer_bubble_sort: NVFP4 ({args.cfg}) quantization done, running fakequant generate")
    else:
        print("infer_bubble_sort: --skip_quant set, running BF16 baseline")

    # ------------------------------------------------------------------
    # 3) 构造 chat prompt
    # ------------------------------------------------------------------
    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    print("=" * 80)
    print(f"PROMPT:\n{args.prompt}")
    print("=" * 80)
    print("RESPONSE (streaming):\n")

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    # ------------------------------------------------------------------
    # 4) 生成
    # ------------------------------------------------------------------
    t0 = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )
    elapsed = time.time() - t0
    new_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    print()
    print("=" * 80)
    print(f"infer_bubble_sort: {new_tokens} tokens in {elapsed:.1f}s "
          f"({new_tokens / max(elapsed, 1e-6):.1f} tok/s)")


if __name__ == "__main__":
    main()

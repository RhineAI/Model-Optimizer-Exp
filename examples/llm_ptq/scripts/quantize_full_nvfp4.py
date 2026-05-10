#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""量化 HuggingFace 模型中的"所有矩阵张量"到 NVFP4.

和 ``hf_ptq.py --qformat=nvfp4`` 的差异:
- 不使用 ``_default_disabled_quantizer_cfg``: 不再默认排除 ``lm_head``,
  ``linear_attn.conv1d``, ``mlp.gate``, ``router``, ``output_layer`` 等
- 不调用 ``extract_and_prepare_language_model_from_vl``: VLM 的 ``visual``
  部分也会参与量化 + 校准
- 不写入 ``_mtp_layer_prefixes``: 推测解码 MTP 层也会被量化
- 校准直接跑 ``full_model`` 而不是 ``language_model``, 确保 vision / MTP
  子模块有激活数据,量化器能估出合理的 amax

只剩下两类模块仍是 BF16:
- 非 Linear/Conv 矩阵模块: ``nn.LayerNorm``, ``nn.RMSNorm``, ``nn.Embedding``,
  ``nn.Conv1d`` (形状 (C,1,k) 这种不是矩阵乘的卷积) 等; NVFP4 量化器只作用在
  Linear/Conv 的 ``*weight_quantizer`` / ``*input_quantizer`` 上
- ``*_bias``: NVFP4 从不量化 bias

所有 ``nn.Linear`` 的 weight / input 都会走 NVFP4.
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.utils.dataset_utils import create_forward_loop, get_dataset_dataloader

# 复用 hf_ptq 示例里的加载器: 它能根据 config.architectures 显式选择正确的
# HF 类 (Qwen3_5ForConditionalGeneration 等), 避免 AutoModelForCausalLM 降级到
# 只有文本塔的 Qwen3_5ForCausalLM 并丢失 architectures 字段
_PTQ_DIR = Path(__file__).resolve().parent.parent
if str(_PTQ_DIR) not in sys.path:
    sys.path.insert(0, str(_PTQ_DIR))
from example_utils import get_model, get_tokenizer  # noqa: E402

RAND_SEED = 1234


def build_full_nvfp4_cfg() -> dict:
    """NVFP4 W4A4, 所有 Linear/Conv 矩阵都量化 (不走 _default_disabled).

    只有 1 条 deny-all + 2 条 wildcard allow, 没有任何"保留层"条目.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyt_ckpt_path", required=True,
                        help="HuggingFace 源模型目录")
    parser.add_argument("--export_path", required=True,
                        help="量化后 HF checkpoint 导出目录")
    parser.add_argument("--dataset", default=None,
                        help="校准数据集名 / 本地 JSONL 路径")
    parser.add_argument("--calib_size", type=int, default=128)
    parser.add_argument("--calib_seq", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise OSError("quantize_full_nvfp4: CUDA GPU required")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)

    mto.enable_huggingface_checkpointing()

    start_time = time.time()

    # ------------------------------------------------------------------
    # 1) 加载模型 + tokenizer (复用 hf_ptq 的 get_model: 它按 architectures
    #    正确选择 Qwen3_5ForConditionalGeneration 等 VLM 类)
    # ------------------------------------------------------------------
    print(f"quantize_full_nvfp4: loading {args.pyt_ckpt_path}")
    full_model = get_model(
        args.pyt_ckpt_path,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    full_model.eval()

    tokenizer = get_tokenizer(
        args.pyt_ckpt_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2) 构建校准 dataloader
    # ------------------------------------------------------------------
    if args.dataset is None:
        raise ValueError("quantize_full_nvfp4: --dataset must be provided "
                         "(either a registered name or a local .jsonl path)")
    print(f"quantize_full_nvfp4: calibration from {args.dataset} "
          f"(n={args.calib_size}, seq={args.calib_seq})")
    calib_dataloader = get_dataset_dataloader(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.calib_size,
        max_sample_length=args.calib_seq,
        device=full_model.device,
    )

    # ------------------------------------------------------------------
    # 3) 跑 NVFP4 量化 (直接作用在 full_model 上; 不剔除 visual / mtp / lm_head)
    # ------------------------------------------------------------------
    quant_cfg = build_full_nvfp4_cfg()
    calibrate_loop = create_forward_loop(dataloader=calib_dataloader)
    print("quantize_full_nvfp4: applying NVFP4 to every weight/input quantizer")
    mtq.quantize(full_model, quant_cfg, forward_loop=calibrate_loop)
    mtq.print_quant_summary(full_model)

    # ------------------------------------------------------------------
    # 4) 导出 HF checkpoint
    # ------------------------------------------------------------------
    export_path = Path(args.export_path)
    export_path.mkdir(parents=True, exist_ok=True)
    print(f"quantize_full_nvfp4: exporting to {export_path}")

    # 故意不设置 _mtp_layer_prefixes: MTP 层要跟普通层一样走量化
    export_hf_checkpoint(full_model, export_dir=export_path)

    tokenizer.save_pretrained(export_path)

    # 补齐 VLM 辅助配置: preprocessor / video / vocab / merges 等 exporter 不写
    src_dir = Path(args.pyt_ckpt_path)
    for name in ("preprocessor_config.json", "video_preprocessor_config.json",
                 "processor_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja"):
        src_file = src_dir / name
        dst_file = export_path / name
        if src_file.is_file() and not dst_file.is_file():
            dst_file.write_bytes(src_file.read_bytes())
            print(f"quantize_full_nvfp4: copied aux file {name}")

    elapsed = time.time() - start_time
    print(f"quantize_full_nvfp4: done in {elapsed:.1f}s -> {export_path}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()

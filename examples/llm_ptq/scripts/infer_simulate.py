#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 fakequant 推理脚本: 让 Qwen3.5-0.8B 写一个冒泡排序.

两种运行模式:
- original:  完全不动模型, 以 HF config 的原始 dtype (Qwen3.5 是 BF16) 加载并推理
- quantized: 先把模型强制转为 FP16, 再对所有 Linear/Conv 的 weight+input 做 NVFP4
             fakequant, 无任何排除项. 非量化张量仍为 FP16

流程:
1) 加载模型 (quantized 模式下再 .half() 到 FP16)
2) 如果是 quantized, 用少量样本做 NVFP4 全量量化校准
3) 直接在当前状态下 forward / generate, 流式打印

为什么走 fakequant 而不是 load 已导出的 NVFP4 checkpoint:
- 导出的 HF checkpoint 是 packed FP4 (uint8 存储), transformers 没有 NVFP4
  的反量化支持, AutoModelForCausalLM 重载时 shape mismatch
- 在本机 (Ampere) 上也没有 NVFP4 tensor core, 只能走 fakequant = FP16 GEMM
  + 每次 matmul 前后 quant/dequant. 这是本项目 examples/vllm_serve/ 同款思路
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import re
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

# 需要 hook 的子模块名片段. 命中任意一项即抓其 forward 输出.
# - input_layernorm / post_attention_layernorm: decoder layer 内的两处 RMSNorm
# - self_attn / self_attn\.(q|k|v|o)_proj: 注意力整体 + 四个投影
# - mlp / mlp\.(gate|up|down)_proj: FFN 整体 + 三个投影
# - model.embed_tokens: 词嵌入
# - model.norm: 最后一层 final RMSNorm
_HOOK_PATTERNS = [
    r"\.embed_tokens$",
    r"model\.norm$",
    r"layers\.\d+$",
    r"layers\.\d+\.input_layernorm$",
    r"layers\.\d+\.post_attention_layernorm$",
    r"layers\.\d+\.self_attn$",
    r"layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
    r"layers\.\d+\.mlp$",
    r"layers\.\d+\.mlp\.(gate|up|down)_proj$",
]
_HOOK_RE = re.compile("|".join(_HOOK_PATTERNS))


def build_full_nvfp4_cfg() -> dict:
    """全量 NVFP4 配置: 所有 Linear/Conv 的 weight + input 全走 NVFP4, 无任何排除项.

    等价于 quantize_full_nvfp4.py 的 quant_cfg.
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


def _tensor_from_output(out):
    """从 module forward 输出里抽出主张量. HF decoder layer 常返回 tuple (hidden, ...)."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def register_dump_hooks(model: torch.nn.Module) -> tuple[dict, list]:
    """给模型中所有命中 _HOOK_RE 的子模块注册 forward hook, 把输出张量 detach 到 CPU.

    返回 (dumps, handles):
      dumps: dict[module_name -> Tensor on cpu, fp16 以省空间]
      handles: hook handles, 调用方 dump 完要逐个 remove
    """
    dumps: dict[str, torch.Tensor] = {}
    handles = []
    hit_names: list[str] = []

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            tensor = _tensor_from_output(output)
            if tensor is None:
                return
            # 只留第一个 batch 的前 N token 会更省, 但这里就是调试用, 全存
            dumps[name] = tensor.detach().to("cpu", dtype=torch.float16).clone()
        return _hook

    for name, module in model.named_modules():
        if _HOOK_RE.search(name):
            handles.append(module.register_forward_hook(_make_hook(name)))
            hit_names.append(name)

    print(f"infer_simulate.register_dump_hooks: installed {len(hit_names)} hooks")
    return dumps, handles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/disk1/guohaoran/models/Qwen3.5-0.8B", help="HF 源模型目录 (BF16)")
    parser.add_argument("--dataset", default="/data/disk1/guohaoran/calib_text_128.jsonl", help="校准数据集 (本地 .jsonl 或已注册名)")
    parser.add_argument("--calib_size", type=int, default=16, help="校准样本数, 推理场景量少即可")
    parser.add_argument("--calib_seq", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--mode", choices=["original", "quantized"], default="quantized", help="original=HF config 原始 dtype (Qwen3.5=BF16) 且不量化; quantized=强制 FP16 + 对所有 Linear/Conv 的 weight+input 做 NVFP4 fakequant, 无排除项")
    parser.add_argument("--dump_states", type=str, default=None, help="若指定, 用该 prompt 跑一次 prefill-only forward, 把每层中间激活 (embed/layernorm/q_proj/.../mlp/logits) dump 成 .pt. 路径为输出文件, 例如 dumps/quantized.pt")
    parser.add_argument("--skip_generate", action="store_true", help="只做 dump, 不跑 generate (配合 --dump_states 使用)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise OSError("infer_simulate: CUDA GPU required")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.manual_seed(RAND_SEED)
    # Qwen3.5 的 linear_attn.conv1d 在未装 causal-conv1d 时走 F.conv1d, 部分
    # cuDNN 版本会抛 CUDNN_STATUS_NOT_INITIALIZED. 关掉 cuDNN 走 ATen 的 CUDA
    # kernel, 对 0.8B 模型无性能影响.
    torch.backends.cudnn.enabled = False
    mto.enable_huggingface_checkpointing()

    # ------------------------------------------------------------------
    # 1) 加载模型 (quantized 模式下强制 FP16; original 保留 HF config 的原始 dtype)
    # ------------------------------------------------------------------
    if args.mode == "quantized":
        print(f"infer_simulate: loading {args.model} and casting to FP16 (quantized mode)")
    else:
        print(f"infer_simulate: loading {args.model} in HF config dtype (original mode)")
    model = get_model(args.model, device="cuda", trust_remote_code=args.trust_remote_code)
    if args.mode == "quantized":
        model = model.half()  # 仅量化路径强制 FP16, 和 NVFP4 fakequant 的 GEMM 数值一致
    model.eval()
    tokenizer = get_tokenizer(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ------------------------------------------------------------------
    # 2) NVFP4 fakequant 校准 (quantized 模式)
    # ------------------------------------------------------------------
    if args.mode == "quantized":
        quant_cfg = build_full_nvfp4_cfg()
        print(f"infer_simulate: calibrating full NVFP4 with {args.calib_size} samples")
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
        print("infer_simulate: NVFP4 quantization done, running FP16+NVFP4 fakequant forward/generate")
    else:
        print("infer_simulate: mode=original, running unmodified baseline")

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

    # ------------------------------------------------------------------
    # 3.5) 可选: dump 每层中间激活 + logits (prefill-only forward)
    # ------------------------------------------------------------------
    if args.dump_states:
        dump_path = Path(args.dump_states)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"infer_simulate.main: dumping intermediate states to {dump_path}")
        dumps, handles = register_dump_hooks(model)
        try:
            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                    return_dict=True,
                )
            # 官方接口: hidden_states (tuple: embed + 每层输出) / attentions (tuple: 每层 attn weights)
            hidden_states = tuple(h.detach().to("cpu", dtype=torch.float16) for h in out.hidden_states)
            attentions = tuple(
                a.detach().to("cpu", dtype=torch.float16) for a in (out.attentions or ()) if a is not None
            )
            payload = {
                "meta": {
                    "mode": args.mode,
                    "model": args.model,
                    "prompt": args.prompt,
                    "input_ids": input_ids.detach().to("cpu"),
                    "attention_mask": attention_mask.detach().to("cpu"),
                },
                "hook_outputs": dumps,  # dict: module_name -> tensor
                "hidden_states": hidden_states,  # tuple: len = num_layers + 1
                "attentions": attentions,
                "logits": out.logits.detach().to("cpu", dtype=torch.float16),
            }
            torch.save(payload, dump_path)
            print(f"infer_simulate.main: dumped {len(dumps)} hook tensors + "
                  f"{len(hidden_states)} hidden_states + {len(attentions)} attentions + logits "
                  f"to {dump_path}")
        finally:
            for h in handles:
                h.remove()

    if args.skip_generate:
        print("infer_simulate.main: --skip_generate set, exiting after dump")
        return

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
    print(f"infer_simulate: {new_tokens} tokens in {elapsed:.1f}s "
          f"({new_tokens / max(elapsed, 1e-6):.1f} tok/s)")


if __name__ == "__main__":
    main()

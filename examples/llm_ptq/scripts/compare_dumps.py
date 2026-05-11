#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""对比 infer_simulate.py 产出的两份 dump.

常见用法: original (HF config 原始 dtype) vs quantized (FP16 + NVFP4 fakequant).

    python examples/llm_ptq/scripts/compare_dumps.py \
        --a examples/llm_ptq/scripts/dumps/original.pt \
        --b examples/llm_ptq/scripts/dumps/quantized.pt

输出三块指标:
1) hidden_states: embed + 每层输出的 MAE / 相对 MAE / max / 余弦相似度
2) logits: MAE / max / 余弦相似度 / top-1 一致率 / 最后一 token top-5
3) hook outputs: 按模块名归类 (数字替换为 N) 取 n / median MAE / max MAE / median cos / min cos
"""

from __future__ import annotations

import argparse
import re

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True, help="baseline dump (.pt)")
    parser.add_argument("--b", required=True, help="target dump   (.pt)")
    return parser.parse_args()


def _cos(x: torch.Tensor, y: torch.Tensor) -> float:
    """两个张量整体展平后的余弦相似度 (float, -1~1)."""
    xf = x.reshape(-1).float()
    yf = y.reshape(-1).float()
    return F.cosine_similarity(xf, yf, dim=0, eps=1e-12).item()


def _cos_per_token(x: torch.Tensor, y: torch.Tensor) -> float:
    """按最后一维做余弦相似度, 其它 batch/seq 维 reduce 成 mean. 更贴近 "每个 token 的语义相似度"."""
    xf = x.float()
    yf = y.float()
    return F.cosine_similarity(xf, yf, dim=-1, eps=1e-12).mean().item()


def main() -> None:
    args = parse_args()
    a = torch.load(args.a, weights_only=False)
    b = torch.load(args.b, weights_only=False)

    mode_a = a["meta"].get("mode", "?")
    mode_b = b["meta"].get("mode", "?")
    n_layers = len(a["hidden_states"]) - 1
    logits_shape = tuple(a["logits"].shape)
    print(f"compare_dumps.main: a.mode={mode_a} b.mode={mode_b} "
          f"n_layers={n_layers} logits_shape={logits_shape}")

    print()
    print("===== hidden_states (embed + each layer output) =====")
    header = f"{'tag':>10s}  {'mae':>10s}  {'rel':>7s}  {'max':>10s}  {'cos_all':>8s}  {'cos_tok':>8s}"
    print(header)
    for i, (ha, hb) in enumerate(zip(a["hidden_states"], b["hidden_states"])):
        fa = ha.float()
        fb = hb.float()
        diff = (fa - fb).abs()
        denom = fa.abs().mean().clamp_min(1e-6)
        tag = "embed" if i == 0 else f"layer{i - 1:02d}"
        cos_all = _cos(fa, fb)
        cos_tok = _cos_per_token(fa, fb)
        print(f"{tag:>10s}  {diff.mean().item():.4e}  "
              f"{(diff.mean() / denom).item():6.2%}  "
              f"{diff.max().item():.4e}  "
              f"{cos_all:8.5f}  {cos_tok:8.5f}")

    print()
    print("===== logits =====")
    la = a["logits"].float()
    lb = b["logits"].float()
    diff = (la - lb).abs()
    cos_all = _cos(la, lb)
    cos_tok = _cos_per_token(la, lb)
    print(f"logits mae={diff.mean().item():.4e} max={diff.max().item():.4e} "
          f"cos_all={cos_all:.5f} cos_per_token={cos_tok:.5f}")

    top_a = la.argmax(-1)
    top_b = lb.argmax(-1)
    agreement = (top_a == top_b).float().mean().item()
    print(f"top-1 agreement per token: {agreement:.3%}")

    last_a = la[0, -1].topk(5)
    last_b = lb[0, -1].topk(5)
    print(f"last-token top5 A ({mode_a:>9s}): "
          f"ids={last_a.indices.tolist()} vals={[f'{v:.2f}' for v in last_a.values.tolist()]}")
    print(f"last-token top5 B ({mode_b:>9s}): "
          f"ids={last_b.indices.tolist()} vals={[f'{v:.2f}' for v in last_b.values.tolist()]}")

    print()
    print("===== hook outputs (按模块名归类) =====")
    groups: dict[str, dict[str, list[float]]] = {}
    for name in a["hook_outputs"]:
        if name not in b["hook_outputs"]:
            continue
        ta = a["hook_outputs"][name].float()
        tb = b["hook_outputs"][name].float()
        if ta.shape != tb.shape:
            continue
        mae = (ta - tb).abs().mean().item()
        cos = _cos(ta, tb)
        tag = re.sub(r"\d+", "N", name)
        slot = groups.setdefault(tag, {"mae": [], "cos": []})
        slot["mae"].append(mae)
        slot["cos"].append(cos)

    header = f"{'module':60s}  {'n':>3s}  {'mae_med':>10s}  {'mae_max':>10s}  {'cos_med':>8s}  {'cos_min':>8s}"
    print(header)
    for tag, slot in groups.items():
        maes = sorted(slot["mae"])
        coss = sorted(slot["cos"])
        n = len(maes)
        mae_med = maes[n // 2]
        mae_max = maes[-1]
        cos_med = coss[n // 2]
        cos_min = coss[0]
        print(f"{tag:60s}  {n:3d}  {mae_med:.4e}  {mae_max:.4e}  "
              f"{cos_med:8.5f}  {cos_min:8.5f}")


if __name__ == "__main__":
    main()

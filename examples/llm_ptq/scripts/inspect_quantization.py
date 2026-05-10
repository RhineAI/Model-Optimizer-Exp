#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""打印 ModelOpt 导出的量化 HuggingFace checkpoint 中每个张量的量化情况.

用法:
    python inspect_quantization.py <model_dir> [--detail linear|all|none] [--csv out.csv]

默认路径指向 /data/disk1/guohaoran/models/Qwen3.5-0.8B-NVFP4-T1.
输出:
    1) 基本信息: hf_quant_config.json 中的量化算法 / group_size / exclude_modules 数
    2) dtype 分布 (按张量数 + 按字节)
    3) 每个 module 的分类表 (Linear/Conv + 是否量化 + bytes + params)
    4) 整体汇总: 量化参数量 / 未量化参数量 / 压缩比估计
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from safetensors import safe_open

# dtype -> 字节数. ModelOpt NVFP4 导出用 U8 存储 packed FP4 (每字节 2 个 FP4 值).
DTYPE_BYTES: dict[str, float] = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "BOOL": 1,
}

# weight 后缀 -> packed FP4 时的"逻辑"字节数 (2 个 FP4 = 1 字节).
FP4_PACKED_SUFFIXES = {"weight"}  # 只针对 U8 存储的 weight


@dataclass
class ModuleInfo:
    """单个 module (以最后一段 key 前缀为界) 聚合信息."""

    name: str
    tensors: dict[str, tuple[tuple[int, ...], str, int]] = field(default_factory=dict)
    # suffix -> (shape, dtype_str, raw_bytes)

    @property
    def suffixes(self) -> tuple[str, ...]:
        return tuple(sorted(self.tensors.keys()))

    @property
    def is_nvfp4(self) -> bool:
        # NVFP4 weight 导出最少需要: weight(U8 打包 FP4) + weight_scale(F8_E4M3) + weight_scale_2(F32)
        # input_scale 仅在激活走静态校准时才写出; NVFP4 W4A4 dynamic 激活不写 input_scale
        req = {"weight", "weight_scale", "weight_scale_2"}
        if not req.issubset(self.tensors.keys()):
            return False
        return self.tensors["weight"][1] == "U8"

    @property
    def is_fp8(self) -> bool:
        # FP8 导出签名: weight(F8_E4M3) + weight_scale(F32 标量) + input_scale(F32)
        if "weight" not in self.tensors or self.tensors["weight"][1] not in {"F8_E4M3", "F8_E5M2"}:
            return False
        return "weight_scale" in self.tensors or "input_scale" in self.tensors

    @property
    def has_bias(self) -> bool:
        return "bias" in self.tensors

    @property
    def is_linear_like(self) -> bool:
        """带 weight 且是 2D (或以上) 的 module, 视为 Linear/Conv."""
        if "weight" not in self.tensors:
            return False
        shape = self.tensors["weight"][0]
        return len(shape) >= 2

    @property
    def weight_numel(self) -> int:
        """逻辑权重参数数. NVFP4 U8 存储打包 2:1, 需还原回逻辑 FP4 元素数."""
        if "weight" not in self.tensors:
            return 0
        shape, dt, _ = self.tensors["weight"]
        n = 1
        for d in shape:
            n *= d
        if self.is_nvfp4:
            # U8 每字节打包 2 个 FP4 值, 逻辑参数数是 packed 数 x 2
            n *= 2
        return n

    @property
    def weight_shape(self) -> tuple[int, ...]:
        return self.tensors["weight"][0] if "weight" in self.tensors else ()

    @property
    def weight_dtype(self) -> str:
        return self.tensors["weight"][1] if "weight" in self.tensors else "-"

    def classify(self, excluded: bool) -> str:
        if self.is_nvfp4:
            return "NVFP4"
        if self.is_fp8:
            return "FP8"
        if self.is_linear_like:
            return "BF16-EXCLUDED" if excluded else "BF16-UNQUANT"
        if self.has_bias or any(s in self.tensors for s in ("A_log", "dt_bias")):
            return "OTHER-PARAM"
        return "NORM/EMBED"

    def logical_bytes(self) -> int:
        """实际磁盘字节数 (不做 FP4 unpack)."""
        return sum(raw for _, _, raw in self.tensors.values())

    def ref_bf16_bytes(self) -> int:
        """假设全部 BF16 存储所需字节 (只算 weight, 用于压缩率对比)."""
        return self.weight_numel * 2


def load_index(model_dir: Path) -> list[Path]:
    """返回需要读取的 safetensors 文件列表, 支持 index.json 分片."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as f:
            idx = json.load(f)
        files = sorted({model_dir / fn for fn in idx["weight_map"].values()})
        return list(files)
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    raise FileNotFoundError(f"inspect_quantization: no safetensors found under {model_dir}")


def load_quant_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "hf_quant_config.json"
    if not cfg_path.is_file():
        return {}
    with cfg_path.open() as f:
        return json.load(f)


def is_excluded(module_name: str, patterns: list[str]) -> bool:
    """modelopt hf_quant_config.exclude_modules 支持 glob."""
    for p in patterns:
        if fnmatch.fnmatchcase(module_name, p):
            return True
        # 处理末尾 * 的子树匹配, 例如 "model.visual*" 要匹配 model.visual.blocks.0.attn.qkv
        if p.endswith("*") and module_name.startswith(p[:-1]):
            return True
    return False


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return int(n * DTYPE_BYTES.get(dtype, 0))


def collect_modules(files: list[Path]) -> dict[str, ModuleInfo]:
    modules: dict[str, ModuleInfo] = {}
    for path in files:
        with safe_open(str(path), framework="pt") as f:
            for key in f.keys():
                slc = f.get_slice(key)
                shape = tuple(slc.get_shape())
                dtype = str(slc.get_dtype())
                nbytes = tensor_bytes(shape, dtype)
                parent, _, suffix = key.rpartition(".")
                mod = modules.setdefault(parent, ModuleInfo(name=parent))
                mod.tensors[suffix] = (shape, dtype, nbytes)
    return modules


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def human_num(n: float) -> str:
    for unit, v in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= v:
            return f"{n / v:.2f}{unit}"
    return f"{n:.0f}"


def print_section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="/data/disk1/guohaoran/models/Qwen3.5-0.8B-NVFP4-T1",
        help="HF 格式量化 checkpoint 目录",
    )
    parser.add_argument(
        "--detail",
        choices=["none", "linear", "all"],
        default="linear",
        help="是否打印每个 module 的明细 (默认只打 Linear/Conv-like)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="把 module 明细写入 CSV 文件",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        print(f"inspect_quantization: directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    quant_cfg = load_quant_config(model_dir)
    exclude_patterns: list[str] = []
    if quant_cfg:
        exclude_patterns = quant_cfg.get("quantization", {}).get("exclude_modules", [])

    files = load_index(model_dir)
    modules = collect_modules(files)

    # ------------------------------------------------------------------
    # 基本信息
    # ------------------------------------------------------------------
    print_section("BASIC")
    print(f"model_dir,{model_dir}")
    print(f"safetensors_files,{len(files)}")
    total_disk = sum(p.stat().st_size for p in files)
    print(f"on_disk_bytes,{total_disk},{human_bytes(total_disk)}")
    if quant_cfg:
        q = quant_cfg.get("quantization", {})
        print(f"producer,{quant_cfg.get('producer', {}).get('name', '-')}"
              f",{quant_cfg.get('producer', {}).get('version', '-')}")
        print(f"quant_algo,{q.get('quant_algo', '-')}")
        print(f"kv_cache_quant_algo,{q.get('kv_cache_quant_algo', '-')}")
        print(f"group_size,{q.get('group_size', '-')}")
        print(f"exclude_modules_count,{len(exclude_patterns)}")
    else:
        print("quant_config,MISSING (not a modelopt-exported checkpoint?)")

    # ------------------------------------------------------------------
    # dtype 分布
    # ------------------------------------------------------------------
    print_section("DTYPE DISTRIBUTION")
    dtype_cnt: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    for m in modules.values():
        for _suf, (shape, dt, nb) in m.tensors.items():
            dtype_cnt[dt] += 1
            dtype_bytes[dt] += nb
    print("dtype,tensors,bytes,human_bytes")
    for dt, cnt in sorted(dtype_cnt.items(), key=lambda x: -x[1]):
        print(f"{dt},{cnt},{dtype_bytes[dt]},{human_bytes(dtype_bytes[dt])}")

    # ------------------------------------------------------------------
    # 每个 module 分类
    # ------------------------------------------------------------------
    per_cat_count: Counter[str] = Counter()
    per_cat_params: Counter[str] = Counter()
    per_cat_bytes: Counter[str] = Counter()
    per_cat_ref_bf16: Counter[str] = Counter()
    rows: list[tuple[str, ...]] = []
    for name in sorted(modules):
        m = modules[name]
        excluded = is_excluded(name, exclude_patterns)
        cat = m.classify(excluded)
        per_cat_count[cat] += 1
        per_cat_params[cat] += m.weight_numel
        per_cat_bytes[cat] += m.logical_bytes()
        per_cat_ref_bf16[cat] += m.ref_bf16_bytes()
        rows.append((
            name,
            cat,
            "x".join(str(d) for d in m.weight_shape) or "-",
            m.weight_dtype,
            str(m.weight_numel),
            ",".join(m.suffixes),
            str(m.logical_bytes()),
        ))

    # ------------------------------------------------------------------
    # 明细表 (受 --detail 控制)
    # ------------------------------------------------------------------
    if args.detail != "none":
        print_section(f"PER-MODULE DETAIL (filter={args.detail})")
        header = ("module", "category", "weight_shape", "weight_dtype",
                  "weight_numel", "suffixes", "bytes")
        print(",".join(header))
        for row in rows:
            cat = row[1]
            if args.detail == "linear" and cat not in {"NVFP4", "FP8", "BF16-UNQUANT", "BF16-EXCLUDED"}:
                continue
            print(",".join(row))

    # ------------------------------------------------------------------
    # 分类汇总
    # ------------------------------------------------------------------
    print_section("CATEGORY SUMMARY")
    print("category,modules,weight_params,module_bytes,human_bytes,vs_BF16_ref,ratio")
    total_params = sum(per_cat_params.values()) or 1
    total_bytes = sum(per_cat_bytes.values()) or 1
    for cat in sorted(per_cat_count, key=lambda c: -per_cat_bytes[c]):
        ratio = per_cat_bytes[cat] / per_cat_ref_bf16[cat] if per_cat_ref_bf16[cat] else 0.0
        print(
            f"{cat},{per_cat_count[cat]},{per_cat_params[cat]},"
            f"{per_cat_bytes[cat]},{human_bytes(per_cat_bytes[cat])},"
            f"{human_bytes(per_cat_ref_bf16[cat])},{ratio:.3f}"
        )
    print(f"TOTAL,{sum(per_cat_count.values())},{total_params},{total_bytes},"
          f"{human_bytes(total_bytes)},-,-")

    # ------------------------------------------------------------------
    # 量化覆盖率
    # ------------------------------------------------------------------
    print_section("QUANTIZATION COVERAGE")
    quantized_cats = {"NVFP4", "FP8"}
    q_params = sum(v for c, v in per_cat_params.items() if c in quantized_cats)
    q_modules = sum(v for c, v in per_cat_count.items() if c in quantized_cats)
    linear_cats = {"NVFP4", "FP8", "BF16-UNQUANT", "BF16-EXCLUDED"}
    linear_params = sum(v for c, v in per_cat_params.items() if c in linear_cats)
    linear_modules = sum(v for c, v in per_cat_count.items() if c in linear_cats)
    print(f"quantized_modules,{q_modules}/{linear_modules}")
    print(f"quantized_linear_params,{human_num(q_params)},{q_params}")
    print(f"linear_params_total,{human_num(linear_params)},{linear_params}")
    cov_modules = q_modules / linear_modules if linear_modules else 0.0
    cov_params = q_params / linear_params if linear_params else 0.0
    print(f"coverage_by_modules,{cov_modules:.3%}")
    print(f"coverage_by_params,{cov_params:.3%}")

    # ------------------------------------------------------------------
    # CSV 输出
    # ------------------------------------------------------------------
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["module", "category", "weight_shape", "weight_dtype",
                             "weight_numel", "suffixes", "bytes"])
            writer.writerows(rows)
        print(f"\ninspect_quantization: wrote CSV -> {args.csv}")


if __name__ == "__main__":
    main()

#!/bin/bash
# Qwen3.5-0.8B NVFP4 全量化脚本 (所有矩阵张量都量化到 NVFP4)
# 用法: bash examples/llm_ptq/scripts/quantize_qwen35_08b_nvfp4_iac.sh
# 前置: 已完成 uv venv + uv pip install -e ".[hf]" + 额外示例依赖
#
# 与 quantize_qwen35_08b_nvfp4.sh 的差异:
# - 不跳过 lm_head / linear_attn.conv1d / visual* / mtp* 这些默认排除项
# - 调用 quantize_full_nvfp4.py (本目录下自写脚本), 绕开 hf_ptq.py 的内置排除
# - 校准跑 full_model (不只跑 language_model), 让 visual / MTP 分支也有激活数据
# - 非量化张量 (LayerNorm/Embedding/bias 等) 强制 FP16, 最终产物只剩
#   FP16 + NVFP4 (packed-uint8 + FP8 block scales) 两种格式, 没有 BF16

set -euo pipefail

# ============================================================================
# 参数(按需修改)
# ============================================================================
SRC_MODEL="/data/disk1/guohaoran/models/Qwen3.5-0.8B"
DST_MODEL="/data/disk1/guohaoran/models/Qwen3.5-0.8B-NVFP4-IAC"
CALIB_JSONL="/data/disk1/guohaoran/calib_text_128.jsonl"
LOG_DIR="/data/disk1/guohaoran/modelopt_logs"
LOG_FILE="${LOG_DIR}/ptq_$(basename "$DST_MODEL").log"

CALIB_SIZE=128
CALIB_SEQ=512
BATCH_SIZE=4
CUDA_DEV=0
DTYPE=fp16

# ============================================================================
# 路径解析 (脚本位于 examples/llm_ptq/scripts/)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTQ_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PTQ_DIR/../.." && pwd)"
VENV="$REPO_ROOT/.venv"

# ============================================================================
# 激活虚拟环境
# ============================================================================
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
else
    echo "quantize_qwen35_08b_nvfp4_iac.sh: venv not found at $VENV" >&2
    echo "请先执行: uv venv --python 3.10 $VENV && source $VENV/bin/activate && uv pip install -e \"$REPO_ROOT[hf]\"" >&2
    exit 1
fi

# ============================================================================
# 生成离线校准 JSONL (文件已存在则跳过)
# ============================================================================
if [ ! -f "$CALIB_JSONL" ]; then
    echo "quantize_qwen35_08b_nvfp4_iac.sh: generating calibration JSONL at $CALIB_JSONL"
    mkdir -p "$(dirname "$CALIB_JSONL")"
    CALIB_JSONL_PATH="$CALIB_JSONL" CALIB_SIZE_PY="$CALIB_SIZE" python - <<'PY'
import json, os, random
random.seed(0)
prompts = [
    "Explain {t} in simple terms with two concrete examples a student can understand.",
    "Write a short technical summary about {t}, covering history and current state of the art.",
    "Describe how {t} works step by step, from basic definition to industry applications.",
    "Compare {t} with related approaches; discuss strengths, weaknesses, and use cases.",
    "Give a balanced overview of {t}, covering theory and real-world deployment at scale.",
]
topics = ["large language models","transformer attention","post-training quantization",
    "mixture of experts","retrieval augmented generation","RLHF","gradient descent",
    "batch normalization","speculative decoding","knowledge distillation","sparse attention",
    "LoRA","chain of thought","tokenization","contrastive learning","diffusion models",
    "NAS","weight pruning","mixed precision training","tensor parallelism","pipeline parallelism",
    "ZeRO optimizer","flash attention","rotary embeddings","graph neural networks",
    "self-supervised learning","vision transformers","image segmentation","object detection",
    "semantic search","vector databases","Bayesian optimization"]
out = os.environ["CALIB_JSONL_PATH"]
n = int(os.environ["CALIB_SIZE_PY"])
with open(out, "w") as f:
    for _ in range(n):
        t = random.choice(prompts).format(t=random.choice(topics)) + " " + \
            random.choice(prompts).format(t=random.choice(topics))
        f.write(json.dumps({"text": t}) + "\n")
print(f"wrote {out} ({os.path.getsize(out)} bytes)")
PY
else
    echo "quantize_qwen35_08b_nvfp4_iac.sh: reusing existing calibration JSONL: $CALIB_JSONL"
fi

# ============================================================================
# 跑全量 NVFP4 PTQ
# ============================================================================
mkdir -p "$LOG_DIR"
echo "quantize_qwen35_08b_nvfp4_iac.sh: quantizing $SRC_MODEL -> $DST_MODEL"
echo "quantize_qwen35_08b_nvfp4_iac.sh: log -> $LOG_FILE"

cd "$PTQ_DIR"
CUDA_VISIBLE_DEVICES="$CUDA_DEV" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -u "$SCRIPT_DIR/quantize_full_nvfp4.py" \
    --pyt_ckpt_path="$SRC_MODEL" \
    --export_path="$DST_MODEL" \
    --dataset="$CALIB_JSONL" \
    --calib_size="$CALIB_SIZE" \
    --calib_seq="$CALIB_SEQ" \
    --batch_size="$BATCH_SIZE" \
    --dtype="$DTYPE" \
    2>&1 | tee "$LOG_FILE"

echo "quantize_qwen35_08b_nvfp4_iac.sh: done. output -> $DST_MODEL"
du -sh "$DST_MODEL"

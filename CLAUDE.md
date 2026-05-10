# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

NVIDIA Model Optimizer (ModelOpt): open-source library for model optimization techniques including
quantization, pruning, distillation, sparsity, and speculative decoding to accelerate inference.
Primarily Python codebase with optional C++/CUDA extensions supporting PyTorch, ONNX, and Hugging Face/Megatron models.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains
> developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**

- NVIDIA Apache 2.0 license header on ALL new Python/C++/CUDA files — use the SPDX format from `LICENSE_HEADER` (auto-inserted by pre-commit for most files, but must be added manually for files copied from third-party sources, which are excluded from the hook)
- `git commit -s -S` (DCO sign-off + cryptographic signing required). Never attribute AI tools in
  sign-off line
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- PRs require CODEOWNERS review (auto-assigned based on `.github/CODEOWNERS`)
- When creating PRs (`gh pr create`), fill in `.github/PULL_REQUEST_TEMPLATE.md` verbatim — do NOT substitute the harness's default `## Summary` / `## Test plan` format
- For non-trivial PRs, run `/claude review` to get Claude approval before merging (NVIDIA org members can self-trigger; orthogonal to CodeRabbit)
- After rebasing, always re-run tests locally before pushing
- All code must follow the security guidelines in `SECURITY.md` — violations are blocked as pre-merge errors
- For contribution guidelines, commit conventions, and PR requirements, see `CONTRIBUTING.md`
- New PIP dependencies require license verification — non-permissive licenses need justification and approval from `@NVIDIA/modelopt-setup-codeowners`

## Common Commands

| Task                      | Command                                                               |
| ------------------------- | --------------------------------------------------------------------- |
| Install (editable + dev)  | `pip install -e ".[dev]"`                                             |
| Enable pre-commit hooks   | `pre-commit install`                                                  |
| CPU unit tests            | `python -m pytest tests/unit`                                         |
| GPU unit tests            | `python -m pytest tests/gpu`                                          |
| Megatron GPU tests        | `python -m pytest tests/gpu_megatron`                                 |
| TRT-LLM GPU tests         | `python -m pytest tests/gpu_trtllm`                                   |
| Single test file          | `python -m pytest tests/unit/torch/quantization/test_quant_config.py` |
| Pattern match             | `pytest tests/unit -k "test_quantize"`                                |
| Lint + format (all files) | `pre-commit run --all-files`                                          |
| Lint (diff only)          | `pre-commit run --from-ref origin/main --to-ref HEAD`                 |
| Run via nox (CPU unit)    | `nox -s "unit-3.12(torch_211, tf_latest)"`                            |
| Build docs                | `nox -s docs`                                                         |
| Build wheel               | `nox -s build_wheel`                                                  |

## Architecture

ModelOpt code base is organized into four top-level namespaces:

| Namespace         | Path               | Role                                                   |
| ----------------- | ------------------ | ------------------------------------------------------ |
| `modelopt.torch`  | `modelopt/torch/`  | Core PyTorch optimization library                      |
| `modelopt.onnx`   | `modelopt/onnx/`   | ONNX model quantization and export                     |
| `modelopt.deploy` | `modelopt/deploy/` | Deployment utilities for LLMs                          |
| `modelopt.recipe` | `modelopt/recipe/` | Recipe loading, parsing, and validation infrastructure |

### `modelopt.torch` Sub-packages

| Sub-package    | Path                           | Role                                                                                                                                                                                                                                              |
| -------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `opt`          | `modelopt/torch/opt/`          | Core optimization infrastructure (modes, config, state dicts)                                                                                                                                                                                     |
| `quantization` | `modelopt/torch/quantization/` | PTQ, QAT, and quantization-aware algorithms                                                                                                                                                                                                       |
| `prune`        | `modelopt/torch/prune/`        | Structured and unstructured pruning                                                                                                                                                                                                               |
| `distill`      | `modelopt/torch/distill/`      | Knowledge distillation                                                                                                                                                                                                                            |
| `sparsity`     | `modelopt/torch/sparsity/`     | Weight and activation sparsity                                                                                                                                                                                                                    |
| `speculative`  | `modelopt/torch/speculative/`  | Speculative decoding (Medusa, EAGLE, etc.)                                                                                                                                                                                                        |
| `nas`          | `modelopt/torch/nas/`          | Neural architecture search                                                                                                                                                                                                                        |
| `export`       | `modelopt/torch/export/`       | Checkpoint export for TRT-LLM / Megatron                                                                                                                                                                                                          |
| `peft`         | `modelopt/torch/peft/`         | QLoRA and PEFT integration                                                                                                                                                                                                                        |
| `kernels`      | `modelopt/torch/kernels/`      | Custom CUDA/Triton kernels grouped by role: `common/attention` (baseline Triton FA), `quantization/{conv,gemm}` (implicit-GEMM CUDA + tensor-quant C++/CUDA + fp4/fp8 Triton), `sparsity/attention` (skip-softmax / N:M / diffusers+LTX backends) |
| `_deploy`      | `modelopt/torch/_deploy/`      | Internal deployment utilities                                                                                                                                                                                                                     |
| `utils`        | `modelopt/torch/utils/`        | Shared utilities and plugin infrastructure                                                                                                                                                                                                        |

### Core Abstraction: Modes

A **mode** is the unit of model optimization in ModelOpt. Each algorithm (quantization, pruning,
etc.) is implemented as one or more modes. Modes are recorded in the model's `modelopt_state` so
optimization workflows can be composed, saved, and restored.

The main entry points are in `modelopt/torch/opt/conversion.py`:

- `apply_mode(model, mode, ...)` — applies an optimization mode to a model
- `restore(model, ...)` — restores a model to a previously saved optimization state
- `save(model, ...)` / `modelopt_state(model)` — captures the current optimization state

### Core Abstraction: Recipes

A **recipe** is a declarative YAML specification of an optimization configuration. Recipes decouple optimization specs from code, enabling reuse, sharing, and version control.

**Built-in recipes** (`modelopt_recipes/`):

- `general/ptq/` — general-purpose PTQ recipes
- `configs/` — shared configuration units referenced by recipes

## Key Files

| File                                           | Role                                                                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `modelopt/torch/opt/mode.py`                   | Base class for all optimization modes                                                                            |
| `modelopt/torch/opt/config.py`                 | Configuration system for modes                                                                                   |
| `modelopt/torch/opt/conversion.py`             | `apply_mode()` / `restore()` entry points                                                                        |
| `modelopt/torch/quantization/__init__.py`      | PTQ/QAT public API                                                                                               |
| `modelopt/torch/export/unified_export_hf.py`   | Unified HF checkpoint export                                                                                     |
| `modelopt/torch/export/model_config_export.py` | TRT-LLM model config export                                                                                      |
| `modelopt/deploy/llm/`                         | LLM deployment utilities                                                                                         |
| `modelopt/recipe/loader.py`                    | `load_recipe()` / `load_config()` public API                                                                     |
| `modelopt/recipe/config.py`                    | Recipe Pydantic models (`ModelOptPTQRecipe`, `RecipeType`)                                                       |
| `modelopt_recipes/general/ptq/`                | Built-in PTQ recipe YAML files                                                                                   |
| `pyproject.toml`                               | Optional dependency groups (`[onnx]`, `[hf]`, `[all]`, `[dev]`); ruff, mypy, pytest, bandit, and coverage config |
| `.pre-commit-config.yaml`                      | Pre-commit hooks (ruff, mypy, clang-format, license headers)                                                     |
| `noxfile.py`                                   | Test session definitions                                                                                         |

## Design Patterns

| Pattern                   | Key Points                                                                                                                |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Mode composition**      | Optimization algorithms are composed as sequences of modes, each recorded in `modelopt_state`                             |
| **Plugin system**         | Optional integrations (HuggingFace, Megatron, etc.) loaded lazily via `import_plugin()`                                   |
| **Optional dependencies** | Features gated by install extras (`[onnx]`, `[hf]`, `[all]`); avoid hard imports at module level                          |
| **Config dataclasses**    | Each mode has a typed config; use Pydantic or dataclass conventions                                                       |
| **State dict**            | Models carry `modelopt_state` for checkpoint save/restore across optimization steps                                       |
| **Declarative recipes**   | YAML-based optimization specs in `modelopt_recipes/`; loaded via `load_recipe()`, passed to the model optimization system |

## CI / Testing

| Layer                     | Location                  | Notes                                                          |
| ------------------------- | ------------------------- | -------------------------------------------------------------- |
| CPU unit tests            | `tests/unit/`             | Fast, no GPU needed; run in pre-merge CI                       |
| GPU unit tests            | `tests/gpu/`              | Requires CUDA GPU                                              |
| Megatron GPU tests        | `tests/gpu_megatron/`     | Requires Megatron-Core + GPU                                   |
| TRT-LLM GPU tests         | `tests/gpu_trtllm/`       | Requires TensorRT-LLM + GPU                                    |
| Example/integration tests | `tests/examples/`         | Integration tests for examples; see `tests/examples/README.md` |
| Pre-commit / lint         | `.pre-commit-config.yaml` | ruff, mypy, clang-format, license headers, bandit              |
| Coverage                  | `pyproject.toml`          | 70% minimum on `modelopt/*`                                    |

## Procedure

### Language

- 交流对话和提问全使用中文
- 代码仅在注释使用中文 代码输出信息和日志等全使用英文
- 提交日志信息和描述用英文

### Code Changes

- 严格规范的类型定义
- 完善的异常处理机制，抛出可能出现的异常，清晰的说明信息
- 打印详细的错误信息以便调试
- 打印日志时，无论何种类型，以`文件名.函数名: `开头，但是抛出异常时不需要
- 保证代码的可读性与可维护性
- 保证代码整洁干净
- 在适当情况下提出优化建议

### Command

- 项目使用`uv`作为包管理器，相关命令优先使用`uv`

### Bug

- 出现问题时，应先彻底分析问题。然后解释Bug出现的的根本原因。最后再提供准确且有针对性的解决方案
- 当你发现错误原因不清晰时，可主动向代码中加入console.log并询问控制台输出，但请在问题解决后移除这些输出
- 若经历多轮调试，反复尝试后，成功修复。应反向分析问题主因，并回退先前不再需要的调试性修改

### Document

- 禁止携带表情和颜文字
- 禁止输出任何框线和箭头的图表，一切用最适合LLM阅读的文本形式
- CSV部分用最密集的形式，禁止任何多余的空格
- 树状信息相关的描述，如文件树，不应该使用符号来描述，直接每一行都写完整的路径加描述

### Function Call

- 写入文件优先使用相对路径，不使用绝对路径
- 路径分隔符统一使用`/`，不要用`\`
- 使用Write工具写入失败时，分成多次写入，每次仅20行左右

## 远程运行

本机 `~/.bashrc` 定义了 `rc` 函数，用于把当前工作目录同步到远程 GPU 服务器并执行命令。

远程环境信息:

- 用户与主机: `guohaoran@10.176.56.244`
- 远程路径: `/data/disk1/guohaoran/<当前目录名>`，例如本项目对应 `/data/disk1/guohaoran/Model-Optimizer-Exp`

同步行为:

- 使用 `rsync -az --chmod=ugo=rwX --info=progress2 --no-inc-recursive`
- 强制排除 `.git/`
- 存在 `.gitignore` 时以 `--filter=':- .gitignore'` 应用忽略规则
- SSH 选项: `-o Compression=no -o ServerAliveInterval=60`
- 仅上行同步(本地 -> 远程)，不会把远程改动拉回本地

用法:

- `rc` 先同步当前目录，然后在远程打开交互式登录 shell，工作目录自动切到远程项目路径
- `rc <命令>` 先同步当前目录，然后在远程项目路径下执行 `<命令>`，命令结束后退出
- `rc -c` 跳过同步，直接打开远程交互式 shell
- `rc -c <命令>` 跳过同步，直接在远程执行 `<命令>`

执行模型:

- `rc <命令>` 等价于 `ssh -i <KEY> -t guohaoran@10.176.56.244 "cd <REMOTE_PROJECT> && <命令>"`，`-t` 分配 TTY，`stderr` 被 `2>/dev/null` 静默以屏蔽 `Connection to ... closed` 提示
- 命令通过 `$@` 拼接后整体作为远程 shell 字符串执行，引号、重定向、管道按本地 shell 展开后再传给远程

使用建议:

- 需要先产生变更再运行时，使用默认形式 `rc <命令>`，保证远程代码是最新的
- 短时间内多次执行同一命令(例如反复跑同一测试),第二次起用 `rc -c <命令>` 以省去同步开销
- 远程产物(权重、日志、结果目录)不会自动拉回本地；如需取回需手动使用 `scp` / `rsync` 从远程路径拉取
- 依赖本地未提交改动时，确认该文件未被 `.gitignore` 排除，否则不会被同步上去

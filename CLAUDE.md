# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

NVIDIA RecSys Examples is a collection of optimized recommender system implementations with focus on large-scale training and inference optimization. The project integrates with PyTorch, TorchRec, Megatron-Core, and includes custom CUDA kernels for performance-critical operations.

**Key Components:**
- **HSTU (Hierarchical Sparse Transformer Unit)**: Ranking/retrieval model with optimized attention kernels, KV-cache management, and inference serving via Triton
- **SID-GR (Semantic-ID based Generative Recommender)**: Semantic-based retrieval model with beam-search decoding
- **DynamicEmb**: Model-parallel dynamic embedding tables with zero-collision hashing, eviction, and admission control
- **RecSys KVCache Manager**: Standalone package for multi-node/multi-tier KV storage with optimized C++ backend

## Codebase Structure

```
recsys-examples/
├── corelib/                  # Core libraries and CUDA kernels
│   ├── hstu/                 # HSTU layer implementation with CUDA kernels
│   ├── dynamicemb/           # Dynamic embedding table implementation
│   ├── gr_decode_atten/      # Beam-search decode attention kernels (CuTe-based)
│   └── recsys_kvcache_manager/  # Standalone KV-cache management package
├── examples/
│   ├── commons/              # Shared utilities, pipeline infrastructure
│   ├── hstu/                 # HSTU training and inference examples
│   │   ├── training/         # Training scripts and benchmarks
│   │   └── inference/        # Inference with KV-cache, Triton Server, C++ AOTInductor
│   ├── sid_gr/               # Semantic-ID retrieval example
│   └── tests/                # Integration tests
└── docker/                   # Docker build configurations
```

## Development Workflow

### Branch Naming
Follow the naming convention: `<type>-<name>` where type is:
- `fea` - New feature
- `enh` - Enhancement to existing feature
- `bug` - Bug fix or regression fix

Use dashes/underscores in the name, not spaces.

### Code Quality Standards

**Naming conventions:**
- Type names (classes, aliases): CamelCase (e.g., `ShardedEmbedding`)
- Variable names: snake_case (e.g., `use_mixed_precision`), private members prefixed with `_` (e.g., `self._plan`)
- Function names: snake_case (e.g., `get_hstu_config()`)

**Linting & Formatting:**
All code must pass pre-commit checks before submitting a PR:
```bash
pre-commit run -a
```

The project uses:
- **isort** (black profile) - Import sorting
- **black** - Code formatting
- **autoflake** - Remove unused imports and variables
- **mypy** - Type checking (excludes: corelib/*, examples/commons/pipeline/*, examples/hstu/ops/triton_ops/*, etc.)
- **codespell** - Spell checking (skips: .git, corelib/hstu/*, third_party/*)

**Commit signing:** All commits must be signed with `-s` flag:
```bash
git commit -s -m "Add cool feature."
```

### Common Commands

**Setup & Installation:**
```bash
# Install in editable mode with dependencies
pip install -e .

# Install specific modules (as needed)
pip install -e corelib/hstu/
pip install -e corelib/dynamicemb/
pip install -e corelib/recsys_kvcache_manager/
pip install -e examples/commons/
```

**Linting & Code Quality:**
```bash
# Run all pre-commit checks
pre-commit run -a

# Run individual checks
isort --profile black .
black .
autoflake --in-place --remove-all-unused-imports --remove-unused-variables .
mypy examples
codespell
```

**Testing:**
```bash
# Run tests (location may vary by module)
python -m pytest examples/tests/

# Run with verbose output
python -m pytest -v examples/tests/
```

**Building CUDA Kernels:**
Some modules (hstu, gr_decode_atten) have Makefiles for building CUDA kernels:
```bash
cd corelib/hstu/
make

cd corelib/gr_decode_atten/
make
```

**Docker:**
```bash
# Build development environment
docker build -f docker/Dockerfile -t recsys-examples:latest .
```

## Key Architectural Patterns

### Module Organization
- **corelib/** modules are standalone packages with their own setup.py and can be installed independently
- **examples/** code uses corelib modules as dependencies
- **examples/commons/** provides shared infrastructure (utilities, pipeline abstractions)

### CUDA Kernel Integration
- HSTU kernels are implemented in corelib/hstu/ with pybind11 bindings
- Attention operations support multiple backends (custom CUDA, CuTe-based)
- Performance-critical paths use fused operations (e.g., layernorm + dropout in HSTU layer)

### KV-Cache Management
- Modern C++ backend in recsys_kvcache_manager/ handles onload/offload with async operation
- Supports compression and multi-node/multi-tier storage
- Benchmarks show latency can be fully hidden under inference

### Dynamic Embeddings
- Zero-collision hashing with eviction and admission control
- Table fusion and capacity sizing aligned to bucket_capacity
- Supports distributed dumping and memory scaling

### Training Infrastructure
- TorchRec integration for large-scale embedding training
- Megatron-Core integration for distributed training optimizations
- Workload-balanced batch shuffling for data parallel training
- Supports pipeline parallelism and sequence parallelism

## Documentation Resources

- **High-level overview**: [README.md](./README.md)
- **Contributing guide**: [CONTRIBUTING.md](./CONTRIBUTING.md)
- **HSTU training**: [examples/hstu/README.md](./examples/hstu/README.md)
- **HSTU inference**: [examples/hstu/inference/README.md](./examples/hstu/inference/README.md)
- **HSTU C++ inference**: [examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md](./examples/hstu/inference/GUIDE_TO_RUN_CPP_INFERENCE_DEMO.md)
- **SID-GR retrieval**: [examples/sid_gr/README.md](./examples/sid_gr/README.md)
- **DynamicEmb**: [corelib/dynamicemb/README.md](./corelib/dynamicemb/README.md)
- **KVCache Manager**: [corelib/recsys_kvcache_manager/README.md](./corelib/recsys_kvcache_manager/README.md)
- **Beam-search attention**: [corelib/gr_decode_atten/README.md](./corelib/gr_decode_atten/README.md)

## Important Notes

- Type checking with mypy is enabled but excludes corelib and certain example subdirectories
- Changes affecting CUDA kernels require testing on appropriate GPU hardware (H100, B200, etc.)
- Documentation in READMEs across the project is comprehensive - refer to relevant guides for specific components
- Release notes in [CHANGELOG.md](./CHANGELOG.md) track feature additions and optimizations by version

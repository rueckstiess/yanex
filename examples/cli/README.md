# Yanex CLI Examples

This directory contains progressively complex examples demonstrating yanex CLI features. Each example focuses on one or two concepts and includes a detailed README.

## Examples

### [01: Basic Usage](01_basic_usage/README.md)
Get started with yanex: running scripts, accessing parameters, logging metrics. Learn the dual-mode pattern (standalone vs tracked).

**Concepts**: `get_param()`, `log_metrics()`, `yanex run`, experiment tracking

---

### [02: Config Files](02_config_files/README.md)
Organize parameters in YAML config files with nested structures and CLI overrides. Understand the parameter hierarchy.

**Concepts**: `--config`, nested parameters, dot notation, parameter precedence

---

### [03: Logging Artifacts](03_logging_artifacts/README.md)
Save experiment outputs as artifacts: CSV files, text reports, matplotlib figures. Access artifacts in experiment directories.

**Concepts**: `copy_artifact()`, `save_artifact()`, `load_artifact()`, automatic format detection, artifact storage

---

### [04: Metadata and Tags](04_metadata_and_tags/README.md)
Organize experiments with names, descriptions, and tags. Filter and search experiments efficiently.

**Concepts**: `--name`, `--description`, `--tag`, filtering, cleanup

---

### [05: Multi-Step Metrics](05_multi_step_metrics/README.md)
Log metrics at each training step/epoch with explicit step tracking. Build metrics incrementally.

**Concepts**: `step` parameter, auto-increment, metric merging, `yanex show`

---

### [06: Bash Integration](06_bash_integration/README.md)
Integrate existing bash scripts and tools. Automatic parameter passing via environment variables.

**Concepts**: `execute_bash_script()`, `YANEX_PARAM_*`, `YANEX_EXPERIMENT_ID`, stdout/stderr capture

---

### [07: Parameter Sweeps](07_parameter_sweeps/README.md)
Run multiple experiments with different parameter combinations automatically. Grid search and parallel execution.

**Concepts**: Comma-separated lists, `range()`, `linspace()`, `logspace()`, cartesian product, `--parallel`

---

### [08: Script CLI Arguments](08_script_cli_arguments/README.md)
Distinguish between yanex parameters (tracked) and script arguments (operational flags). Use argparse with yanex.

**Concepts**: `--param` vs script args, `--` separator, argparse integration

---

### [09: Staged Experiments](09_staged_experiments/README.md)
Prepare experiments now, execute later. Batch processing and parallel execution of staged experiments.

**Concepts**: `--stage`, `yanex run --staged`, `--parallel`, deferred execution

---

### [10: Dependencies and Multi-Stage Pipelines](10_dependencies/README.md)
Build multi-stage ML pipelines with experiment dependencies. Track complete data lineage and reuse preprocessing across training runs.

**Concepts**: `-D` / `--depends-on`, dependency sweeps, cartesian products, incremental staging, `get_dependencies()`, artifact sharing

---

## Learning Path

**New to yanex?** Start with examples in order:
1. **01-04**: Core features (parameters, config, artifacts, metadata)
2. **05-07**: Training and scaling (multi-step metrics, bash, sweeps)
3. **08-10**: Advanced features (script args, staging, dependencies)

**Quick references:**
- **ML/Training workflows**: 02 → 05 → 07 → 10
- **Multi-stage pipelines**: 03 → 10
- **Bash tool integration**: 06 → 08
- **Batch processing**: 07 → 09 → 10
- **Experiment organization**: 04 → 07 → 10

## Running Examples

Each example can be run standalone or with yanex:

```bash
# Standalone (uses defaults)
python 01_basic_usage/compute.py

# With yanex (tracked)
yanex run 01_basic_usage/compute.py -p num_items=5000

# With config file
yanex run 02_config_files/train.py --config 02_config_files/train-config.yaml
```

See individual README files for detailed usage instructions and expected output.

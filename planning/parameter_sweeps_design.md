# Parameter Sweeps Design Document

## Overview

This document outlines the design and implementation plan for parameter sweep functionality in Yanex.

## Design Decisions

### 1. CLI-Only Feature
- Parameter sweeps are **CLI-only** convenience features
- Python API remains clean and simple - users write their own loops with NumPy/Python
- This keeps the core API focused on single experiment creation

### 2. Execution Strategy
- Parameter sweeps **require explicit `--stage` flag**
- Without `--stage`, sweep syntax triggers an error to prevent accidental batch execution
- All sweep experiments are created as staged experiments
- Users execute with `yanex run --staged` when ready

### 3. Sweep Syntax
Support four sweep types in CLI `--param` arguments:

- `range(start, stop, step)` - Python-like range function
- `linspace(start, stop, count)` - NumPy-like linear spacing
- `logspace(start, stop, count)` - NumPy-like logarithmic spacing  
- `list(item1, item2, ...)` - Explicit list of values (supports mixed types)

### 4. Cross-Product Behavior
- Multiple sweep parameters create cross-product combinations
- Each combination becomes a separate staged experiment
- Example: 5 learning rates × 3 batch sizes = 15 staged experiments

## Implementation Plan

### Phase 1: Sweep Parameter Detection and Parsing ✅
- [x] Extend `_parse_parameter_value()` in `yanex/core/config.py`
- [x] Add sweep syntax detection (regex patterns)
- [x] Implement sweep value generation functions
- [x] Return special sweep objects vs. regular values

### Phase 2: Sweep Expansion Logic ✅
- [x] Create `expand_parameter_sweeps()` function
- [x] Generate all cross-product combinations
- [x] Handle mixed sweep/regular parameters
- [x] Validate sweep parameter constraints

### Phase 3: CLI Integration ✅
- [x] Detect sweep parameters in run command
- [x] Enforce `--stage` requirement for sweeps
- [x] Create multiple staged experiments from combinations
- [x] Add clear error messages for missing `--stage`

### Phase 4: Testing and Documentation ✅
- [x] Unit tests for all sweep types (34 tests)
- [x] Integration tests for CLI behavior (8 tests)
- [x] Cross-product combination tests
- [x] Update CLI help and examples

## Example Usage

```bash
# Single parameter sweep (requires --stage)
yanex run train.py --param "learning_rate=linspace(0.001, 0.1, 5)" --stage

# Multi-parameter sweep (cross-product)
yanex run train.py \
  --param "learning_rate=range(0.01, 0.1, 0.01)" \
  --param "batch_size=list(16, 32, 64)" \
  --stage

# Mixed sweep and regular parameters
yanex run train.py \
  --param "learning_rate=linspace(0.001, 0.1, 5)" \
  --param "model_type=resnet50" \
  --stage

# Execute all staged experiments
yanex run --staged
```

## Error Cases

```bash
# Error: sweep without --stage
yanex run train.py --param "lr=range(0.01, 0.1, 0.01)"
# Error: Parameter sweeps require --stage flag to avoid accidental batch execution

# Error: invalid sweep syntax
yanex run train.py --param "lr=range(0.01)" --stage
# Error: range() requires 3 arguments: start, stop, step
```

## Python API Alternative

Users requiring programmatic sweeps use standard Python/NumPy:

```python
import numpy as np
from yanex import ExperimentManager

manager = ExperimentManager()

# User-controlled loops
for lr in np.linspace(0.001, 0.1, 5):
    for batch_size in [16, 32, 64]:
        manager.create_experiment(
            script_path="train.py",
            config={"learning_rate": lr, "batch_size": batch_size},
            stage_only=True
        )

# Execute staged experiments
manager.execute_staged_experiments()
```

This approach keeps the API clean while providing CLI convenience for common sweep patterns.
# Artifacts API Changes

**Status:** Planning - Implementation Ready
**Date:** 2025-11-23
**Context:** Redesign artifact logging/loading API with automatic format detection and intuitive naming

## Background

### Current Limitation

The current artifact API has several issues:

1. **Unintuitive naming** - `log_artifact()` copies files (not clear from name)
2. **Type-specific functions** - Separate functions for different types (`log_matplotlib_figure()`, `log_text()`)
3. **Manual loading** - No unified loading API, users write custom code for each format
4. **No format detection** - Must know format in advance, no automatic handling
5. **Verbose patterns** - Common operations require boilerplate code

### Current API

**Run API (`yanex.*`):**
```python
# Logging artifacts
yanex.log_artifact(src_path)           # Copies file (unintuitive name)
yanex.log_matplotlib_figure(fig, name) # Matplotlib-specific
yanex.log_text(content, filename)      # Text-specific
```

**Results API (`yanex.results.Experiment`):**
```python
# Loading artifacts - manual approach
exp = manager.get_experiment("exp_id")
artifact_path = exp.get_artifact("model.pt")  # Returns Path
model = torch.load(artifact_path)              # Manual loading

all_paths = exp.get_artifacts()  # Returns list[Path]
```

### Use Case

Common ML workflow with various artifact types:

```python
# Training script - current approach (verbose)
import yanex
import torch
import matplotlib.pyplot as plt
import json

torch.save(model.state_dict(), "model_temp.pt")
yanex.log_artifact("model_temp.pt")  # Unintuitive: why save then copy?

fig, ax = plt.subplots()
ax.plot(losses)
yanex.log_matplotlib_figure(fig, "training_loss.png")  # Type-specific

with open("results_temp.json", "w") as f:
    json.dump(results, f)
yanex.log_artifact("results_temp.json")  # More temp files!

# Evaluation script - current approach (manual loading)
from yanex.results import ResultsManager
manager = ResultsManager()
exp = manager.get_experiment("train_id")

model_path = exp.get_artifact("model.pt")
model.load_state_dict(torch.load(model_path))  # Manual loading

results_path = exp.get_artifact("results.json")
with open(results_path) as f:
    results = json.load(f)  # Manual loading
```

**Goal:** Unified, intuitive API with automatic format detection:

```python
# Training script - proposed API
import yanex

# Save Python objects directly (no temp files)
yanex.save_artifact(model.state_dict(), "model.pt")  # Auto-detects torch
yanex.save_artifact(fig, "training_loss.png")        # Auto-detects matplotlib
yanex.save_artifact(results, "results.json")         # Auto-detects JSON

# Copy existing files when needed
yanex.copy_artifact("data/raw.csv", "raw_data.csv")

# Evaluation script - proposed API (auto-loading)
from yanex.results import ResultsManager
manager = ResultsManager()
exp = manager.get_experiment("train_id")

model_state = exp.load_artifact("model.pt")  # Returns loaded object
results = exp.load_artifact("results.json")  # Returns parsed dict
```

## Design Decisions

### 1. Clean Break - Remove Old API, Introduce New Functions

**Removed functions (breaking change in v1.0.0):**

**Run API:**
- `log_artifact(src_path)` → REMOVED, use `copy_artifact()`
- `log_matplotlib_figure(fig, name)` → REMOVED, use `save_artifact()`
- `log_text(content, filename)` → REMOVED, use `save_artifact()`

**Results API:**
- `exp.get_artifact(filename)` → REMOVED, use `exp.load_artifact()` or `exp.artifacts_dir / filename`
- `exp.get_artifacts()` → REMOVED, use `exp.list_artifacts()`

**New API (Run API):**
- `copy_artifact(src_path, filename=None)` - Copy existing files
- `save_artifact(obj, filename, saver=None)` - Save Python objects with auto-detection
- `load_artifact(filename, loader=None)` - Load with auto-detection

**New API (Results API):**
- `exp.load_artifact(filename, loader=None)` - Load with auto-detection
- `exp.artifact_exists(filename)` - Check existence
- `exp.list_artifacts()` - List artifact filenames

**Rationale:**
- Clear naming: `copy_artifact` = copy file, `save_artifact` = serialize object
- Unified interface reduces API surface
- Auto-detection eliminates type-specific functions
- Clean break acceptable in 0.x → 1.0 transition
- Consistent API across Run and Results interfaces

### 2. copy_artifact() - Copy Existing Files

Copy an existing file to the experiment's artifacts directory:

```python
def copy_artifact(src_path: Path | str, filename: str | None = None) -> None:
    """
    Copy an existing file to the experiment's artifacts directory.

    Args:
        src_path: Path to source file
        filename: Name to use in artifacts dir (defaults to source filename)

    Examples:
        # Copy with same name
        yanex.copy_artifact("data/results.csv")

        # Copy with different name
        yanex.copy_artifact("output.txt", "final_output.txt")
    """
```

**Behavior:**
- **With yanex tracking:** Copy to `~/.yanex/experiments/{exp_id}/artifacts/{filename}`
- **Standalone mode:** Copy to `./artifacts/{filename}` (creates dir if needed)
- Overwrites existing artifact with same name
- Raises `FileNotFoundError` if source doesn't exist

**Rationale:**
- Clear name indicates file copy operation
- Useful for large files already on disk (datasets, pre-trained models)
- Standalone mode enables local testing without yanex tracking

### 3. save_artifact() - Save Python Objects with Auto-Detection

Save a Python object to artifacts directory with automatic format detection:

```python
def save_artifact(
    obj: Any,
    filename: str,
    saver: Callable[[Any, Path], None] | None = None
) -> None:
    """
    Save a Python object to the experiment's artifacts directory.
    Format is auto-detected from filename extension.

    Args:
        obj: Python object to save
        filename: Name for saved artifact (extension determines format)
        saver: Optional custom saver function (obj, path) -> None

    Supported formats (auto-detected):
        .txt        - Plain text (str.write)
        .csv        - CSV (pandas.DataFrame.to_csv or list of dicts)
        .json       - JSON (json.dump)
        .jsonl      - JSON Lines (one JSON object per line)
        .npy        - NumPy array (numpy.save)
        .npz        - NumPy arrays (numpy.savez)
        .pt, .pth   - PyTorch (torch.save)
        .pkl        - Pickle (pickle.dump)
        .png        - Matplotlib figure (fig.savefig)

    Examples:
        # Text
        yanex.save_artifact("Training complete", "status.txt")

        # JSON
        yanex.save_artifact({"acc": 0.95}, "metrics.json")

        # PyTorch model
        yanex.save_artifact(model.state_dict(), "model.pt")

        # Matplotlib figure
        yanex.save_artifact(fig, "plot.png")

        # Custom format
        def save_custom(obj, path):
            with open(path, 'wb') as f:
                custom_serialize(obj, f)

        yanex.save_artifact(my_obj, "data.custom", saver=save_custom)
    """
```

**Format Auto-Detection Rules:**

| Extension | Type Detection | Saver |
|-----------|---------------|-------|
| `.txt` | `isinstance(obj, str)` | `path.write_text(obj)` |
| `.csv` | `isinstance(obj, pd.DataFrame)` | `obj.to_csv(path)` |
| `.csv` | `isinstance(obj, list)` + dict items | `csv.DictWriter` |
| `.json` | Any JSON-serializable | `json.dump(obj, f)` |
| `.jsonl` | `isinstance(obj, list)` | Write one JSON per line |
| `.npy` | `isinstance(obj, np.ndarray)` | `np.save(path, obj)` |
| `.npz` | `isinstance(obj, dict)` + numpy arrays | `np.savez(path, **obj)` |
| `.pt`, `.pth` | Any (torch installed) | `torch.save(obj, path)` |
| `.pkl` | Any (fallback) | `pickle.dump(obj, f)` |
| `.png` | `isinstance(obj, matplotlib.figure.Figure)` | `obj.savefig(path)` |

**Behavior:**
- **With yanex tracking:** Save to `~/.yanex/experiments/{exp_id}/artifacts/{filename}`
- **Standalone mode:** Save to `./artifacts/{filename}` (creates dir if needed)
- Overwrites existing artifact with same name
- Raises `ValueError` if format can't be auto-detected and no custom saver provided
- Raises `ImportError` if required library not installed (e.g., torch, pandas)

**Rationale:**
- Single function for all types reduces cognitive load
- Extension-based detection is intuitive (`.json` → JSON)
- Custom saver allows arbitrary formats
- Standalone mode makes scripts testable locally

### 4. load_artifact() - Load with Auto-Detection

Load an artifact with automatic format detection:

**Run API:**
```python
def load_artifact(
    filename: str,
    loader: Callable[[Path], Any] | None = None
) -> Any | None:
    """
    Load an artifact from the current experiment with automatic format detection.
    Returns None if artifact doesn't exist (allows optional artifacts).

    Args:
        filename: Name of artifact to load
        loader: Optional custom loader function (path) -> object

    Supported formats (auto-detected by extension):
        .txt        - Plain text (returns str)
        .csv        - CSV (returns pandas.DataFrame or list[dict])
        .json       - JSON (returns parsed dict/list)
        .jsonl      - JSON Lines (returns list[dict])
        .npy        - NumPy array (returns np.ndarray)
        .npz        - NumPy arrays (returns dict of arrays)
        .pt, .pth   - PyTorch (returns loaded object)
        .pkl        - Pickle (returns unpickled object)
        .png        - Image (returns PIL.Image)

    Examples:
        # Load from current experiment
        model_state = yanex.load_artifact("model.pt")
        results = yanex.load_artifact("results.json")

        # Optional artifact (returns None if missing)
        checkpoint = yanex.load_artifact("checkpoint.pt")
        if checkpoint is not None:
            model.load_state_dict(checkpoint)

        # Custom loader
        def load_custom(path):
            with open(path, 'rb') as f:
                return custom_deserialize(f)

        obj = yanex.load_artifact("data.custom", loader=load_custom)
    """
```

**Results API:**
```python
# Same signature, different context (loads from specific experiment)
exp.load_artifact(filename: str, loader: Callable[[Path], Any] | None = None) -> Any | None
```

**Format Auto-Detection Rules:**

| Extension | Loader | Return Type |
|-----------|--------|-------------|
| `.txt` | `path.read_text()` | `str` |
| `.csv` | `pd.read_csv(path)` | `pd.DataFrame` |
| `.json` | `json.load(f)` | `dict` or `list` |
| `.jsonl` | Parse line-by-line | `list[dict]` |
| `.npy` | `np.load(path)` | `np.ndarray` |
| `.npz` | `np.load(path)` | `dict[str, np.ndarray]` |
| `.pt`, `.pth` | `torch.load(path)` | Depends on saved object |
| `.pkl` | `pickle.load(f)` | Depends on saved object |
| `.png` | `PIL.Image.open(path)` | `PIL.Image.Image` |

**Behavior:**

**Run API:**
- **With yanex tracking:** Load from `~/.yanex/experiments/{exp_id}/artifacts/{filename}`
- **Standalone mode:** Load from `./artifacts/{filename}`
- Returns `None` if artifact doesn't exist

**Results API:**
- Load from `~/.yanex/experiments/{exp_id}/artifacts/{filename}` (specific experiment)
- Returns `None` if artifact doesn't exist

**Shared behavior:**
- Raises `ValueError` if format can't be auto-detected and no custom loader provided
- Raises `ImportError` if required library not installed

**Rationale:**
- Symmetric with `save_artifact()` - same extension rules
- `None` return for missing artifacts supports optional patterns
- Custom loader handles arbitrary formats
- Consistent behavior across Run and Results APIs

### 5. Standalone Mode Behavior

When running scripts without yanex tracking (`python script.py` instead of `yanex run script.py`):

**Directory:** All artifact operations use `./artifacts/` folder in current working directory

**Behavior:**
```python
# script.py (run as: python script.py)

# Save creates ./artifacts/ if needed
yanex.save_artifact(model, "model.pt")  # Saves to ./artifacts/model.pt
yanex.copy_artifact("data.csv")          # Copies to ./artifacts/data.csv

# Load from local artifacts
model = yanex.load_artifact("model.pt")  # Loads from ./artifacts/model.pt
```

**Rationale:**
- Makes scripts testable without yanex infrastructure
- Consistent behavior: always save/load from artifacts directory
- `./artifacts/` is intuitive and commonly used in ML projects
- `.gitignore` typically excludes `artifacts/` folder

### 6. Helper Functions

Additional utilities for artifact management:

**Run API:**
```python
def artifact_exists(filename: str) -> bool:
    """
    Check if an artifact exists without loading it.

    Args:
        filename: Name of artifact

    Returns:
        True if artifact exists, False otherwise

    Examples:
        if yanex.artifact_exists("checkpoint.pt"):
            model.load_state_dict(yanex.load_artifact("checkpoint.pt"))
    """

def list_artifacts() -> list[str]:
    """
    List all artifacts in the current experiment.

    Returns:
        List of artifact filenames

    Examples:
        artifacts = yanex.list_artifacts()
        # Returns: ["model.pt", "metrics.json", "plot.png"]
    """
```

**Keep existing:**
```python
yanex.get_artifacts_dir() -> Path | None
    """Get path to current experiment's artifacts directory."""
```

**Results API:**
```python
# Same methods, different context (specific experiment)
exp.artifact_exists(filename: str) -> bool
exp.list_artifacts() -> list[str]
```

**Keep existing:**
```python
exp.artifacts_dir -> Path  # Property
    """Get path to experiment's artifacts directory."""
```

**Rationale:**
- `artifact_exists()` avoids loading for existence checks
- `list_artifacts()` returns filenames (not paths) - simpler, more common use case
- `artifacts_dir` property/method provides path for manual construction: `exp.artifacts_dir / filename`
- Consistent API across Run and Results interfaces

### 7. Results API Integration

The Results API mirrors the Run API for consistency:

**New methods on `Experiment` class:**
```python
# Auto-loading with format detection
model = exp.load_artifact("model.pt")  # Returns loaded object
data = exp.load_artifact("data.json")  # Returns parsed dict

# Check existence
if exp.artifact_exists("checkpoint.pt"):
    checkpoint = exp.load_artifact("checkpoint.pt")

# List all artifacts
artifacts = exp.list_artifacts()  # Returns: ["model.pt", "data.json", "plot.png"]
```

**Keep existing property:**
```python
# Get artifacts directory path (for manual path construction)
model_path = exp.artifacts_dir / "model.pt"
with open(model_path, 'rb') as f:
    model = torch.load(f)
```

**Removed methods (clean break):**
- `exp.get_artifact(filename)` → Use `exp.load_artifact()` or `exp.artifacts_dir / filename`
- `exp.get_artifacts()` → Use `exp.list_artifacts()`

**Complete example:**
```python
from yanex.results import ResultsManager

manager = ResultsManager()
exp = manager.get_experiment("abc123")

# New API - auto-loading
model_state = exp.load_artifact("model.pt")
results = exp.load_artifact("results.json")

# List artifacts
for artifact_name in exp.list_artifacts():
    print(f"Found artifact: {artifact_name}")

# Path access when needed (escape hatch)
custom_path = exp.artifacts_dir / "custom_file.bin"
if custom_path.exists():
    with open(custom_path, 'rb') as f:
        data = f.read()
```

**Rationale:**
- Consistent API across Run and Results interfaces
- Auto-loading eliminates boilerplate in analysis scripts
- Clean break acceptable in 0.x versions
- `artifacts_dir` property provides escape hatch for path-based tools

### 8. Migration and Breaking Changes

**Breaking changes in v1.0.0:**

**Run API:**
- `log_artifact()` → REMOVED, use `copy_artifact()`
- `log_matplotlib_figure()` → REMOVED, use `save_artifact()`
- `log_text()` → REMOVED, use `save_artifact()`

**Results API:**
- `exp.get_artifact(filename)` → REMOVED, use `exp.load_artifact()` or `exp.artifacts_dir / filename`
- `exp.get_artifacts()` → REMOVED, use `exp.list_artifacts()`

**Migration guide:**

```python
# OLD: Run API
yanex.log_artifact("model.pt")
yanex.log_text(content, "output.txt")
yanex.log_matplotlib_figure(fig, "plot.png")

# NEW: Run API
yanex.copy_artifact("model.pt")
yanex.save_artifact(content, "output.txt")
yanex.save_artifact(fig, "plot.png")

# OLD: Results API
model_path = exp.get_artifact("model.pt")
model = torch.load(model_path)
all_paths = exp.get_artifacts()

# NEW: Results API
model = exp.load_artifact("model.pt")  # Auto-loads!
all_names = exp.list_artifacts()       # Returns filenames

# Path access still possible
model_path = exp.artifacts_dir / "model.pt"
model = torch.load(model_path)
```

**Rationale:**
- Clean break acceptable in 0.x → 1.0 transition
- Simpler codebase (no deprecated code paths)
- Users can update in single session
- Clear migration path with 1:1 replacements

### 9. Error Handling and Validation

**File extension validation:**
```python
# If extension not recognized and no custom saver/loader
raise ValueError(
    f"Cannot auto-detect format for '{filename}'. "
    f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}. "
    f"Use 'saver' parameter for custom formats."
)
```

**Missing dependencies:**
```python
# If format requires library not installed
raise ImportError(
    f"Loading {filename} requires {library_name}. "
    f"Install with: pip install {library_name}"
)
```

**Overwrite behavior:**
- Overwrites existing artifacts with same name (no warning)
- Rationale: Simplifies workflow, matches expected behavior

**Rationale:**
- Clear error messages guide users to solutions
- Fail fast with actionable feedback
- Overwriting is expected behavior in iterative development

## Implementation Overview

### Core Modules to Create/Modify

**New modules:**
1. `yanex/core/artifact_formats.py` - Format detection and handlers
2. `yanex/core/artifact_io.py` - Unified save/load logic

**Modified modules:**
1. `yanex/api.py` - Remove old functions, add new ones
2. `yanex/core/storage_artifacts.py` - Unified storage interface
3. `yanex/results/experiment.py` - Remove old methods, add new ones

### Format Handler Architecture

```python
# yanex/core/artifact_formats.py

from dataclasses import dataclass
from typing import Callable, Any
from pathlib import Path

@dataclass
class FormatHandler:
    """Handler for a specific artifact format"""
    extensions: list[str]
    type_check: Callable[[Any], bool]  # Check if object matches format
    saver: Callable[[Any, Path], None]
    loader: Callable[[Path], Any]

# Registry of format handlers
FORMAT_HANDLERS = [
    FormatHandler(
        extensions=[".json"],
        type_check=lambda obj: True,  # JSON handles many types
        saver=lambda obj, path: json.dump(obj, open(path, 'w')),
        loader=lambda path: json.load(open(path))
    ),
    FormatHandler(
        extensions=[".pt", ".pth"],
        type_check=lambda obj: True,
        saver=lambda obj, path: torch.save(obj, path),
        loader=lambda path: torch.load(path)
    ),
    # ... more handlers
]

def get_handler_for_save(obj: Any, filename: str) -> FormatHandler:
    """Find appropriate handler based on object type and filename"""
    ext = Path(filename).suffix.lower()
    for handler in FORMAT_HANDLERS:
        if ext in handler.extensions and handler.type_check(obj):
            return handler
    raise ValueError(f"No handler found for {filename}")

def get_handler_for_load(filename: str) -> FormatHandler:
    """Find appropriate handler based on filename extension"""
    ext = Path(filename).suffix.lower()
    for handler in FORMAT_HANDLERS:
        if ext in handler.extensions:
            return handler
    raise ValueError(f"No handler found for {filename}")
```

### Standalone Mode Detection

```python
# In yanex/api.py or similar

def _get_artifacts_dir() -> Path:
    """Get artifacts directory (current experiment or standalone)"""
    if is_tracking_enabled():
        exp_id = get_current_experiment_id()
        return Path(f"~/.yanex/experiments/{exp_id}/artifacts").expanduser()
    else:
        # Standalone mode - use local artifacts folder
        artifacts_dir = Path("./artifacts")
        artifacts_dir.mkdir(exist_ok=True)
        return artifacts_dir
```

## API Reference

### Run API (`yanex.*`)

```python
# Saving artifacts
yanex.copy_artifact(src_path: Path | str, filename: str | None = None) -> None
yanex.save_artifact(obj: Any, filename: str, saver: Callable | None = None) -> None

# Loading artifacts (current experiment)
yanex.load_artifact(filename: str, loader: Callable | None = None) -> Any | None

# Helper functions
yanex.artifact_exists(filename: str) -> bool
yanex.list_artifacts() -> list[str]
yanex.get_artifacts_dir() -> Path | None
```

### Results API (`yanex.results.Experiment`)

```python
# Loading artifacts
exp.load_artifact(filename: str, loader: Callable | None = None) -> Any | None

# Helper functions
exp.artifact_exists(filename: str) -> bool
exp.list_artifacts() -> list[str]
exp.artifacts_dir -> Path  # Property
```

## Scope Clarifications

**Included in this feature:**
- ✅ New unified artifact API (`copy_artifact`, `save_artifact`, `load_artifact`)
- ✅ Automatic format detection for common types
- ✅ Custom saver/loader support for arbitrary formats
- ✅ Standalone mode using `./artifacts/` directory
- ✅ Helper functions (exists, list)
- ✅ Results API integration with same methods
- ✅ Clean break from old API (no backward compatibility)
- ✅ Clear error messages for unsupported formats

**Explicitly OUT OF SCOPE** (future features):
- ❌ Artifact versioning or history tracking
- ❌ Remote artifact storage (S3, GCS, etc.)
- ❌ Lazy loading or streaming for large artifacts
- ❌ Artifact compression or encryption
- ❌ Dependency-aware artifact loading (separate feature)

## References

- **Similar systems:** MLflow (artifacts), Weights & Biases (artifacts), DVC (artifact tracking)
- **Python libraries:** pickle, json, torch.save/load, numpy.save/load, pandas.to_csv/read_csv
- **Format detection:** Based on file extensions (industry standard)

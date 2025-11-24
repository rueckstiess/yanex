# Custom Artifact Handlers Design

## Problem Statement

Users need to save/load custom object types (e.g., domain-specific data structures like Workloads, Cache objects) that have their own serialization methods. Currently, they must:

1. Use verbose custom saver/loader on every call
2. Manually create paths and call object methods
3. Repeat the same saver/loader logic throughout codebase

**Current workaround:**
```python
# Verbose - repeated everywhere
yanex.save_artifact(train, "workload_train.jsonl", saver=lambda obj, path: obj.save(str(path)))

# Or manual approach
train_path = Path("../artifacts/workload_train.jsonl").resolve()
train.save(str(train_path))
yanex.log_artifact(name="workload_train.jsonl", file_path=train_path)
```

**Desired API:**
```python
# Clean - register once, use everywhere
yanex.save_artifact(train, "workload_train.jsonl")  # Auto-detects Workload type
loaded = yanex.load_artifact("workload_train.jsonl", format="workload")
```

## Current Implementation

Yanex has an internal format handler registry in `yanex/core/artifact_formats.py`:

```python
@dataclass
class FormatHandler:
    extensions: list[str]
    type_check: Callable[[Any], bool]
    saver: Callable[[Any, Path], None]
    loader: Callable[[Path], Any]
    required_package: str | None = None

FORMAT_HANDLERS = [
    FormatHandler(
        extensions=[".txt"],
        type_check=lambda obj: isinstance(obj, str),
        saver=_save_text,
        loader=_load_text,
    ),
    # ... more handlers for .json, .jsonl, .pkl, .pt, .npy, .csv, .png
]
```

**Limitation:** Registry is internal and not modifiable by users.

## Proposed Solution

Add a public registration API that allows users to register custom format handlers with:

1. **Named formats** - Explicit format identifier for loading ambiguous extensions
2. **Type-based auto-detection** - Automatic handler selection during save
3. **Shared extensions** - Multiple handlers can handle same extension (e.g., `.jsonl`)
4. **Backwards compatibility** - Existing code continues to work

## API Design

### 1. Registration API

```python
def register_format(
    name: str,
    extensions: list[str],
    type_check: Callable[[Any], bool],
    saver: Callable[[Any, Path], None],
    loader: Callable[[Path], Any],
    required_package: str | None = None,
) -> None:
    """Register a custom artifact format handler.

    Args:
        name: Format identifier for explicit loading (e.g., "workload")
        extensions: File extensions this handler supports (e.g., [".jsonl"])
        type_check: Function to check if object matches this handler
        saver: Function to save object to path: (obj, path) -> None
        loader: Function to load object from path: (path) -> object
        required_package: Optional package name required for this handler

    Examples:
        # Register workload format
        yanex.register_format(
            name="workload",
            extensions=[".jsonl"],
            type_check=lambda obj: isinstance(obj, Workload),
            saver=lambda obj, path: obj.save(str(path)),
            loader=lambda path: Workload.load(str(path)),
        )

        # Register format requiring optional dependency
        yanex.register_format(
            name="parquet",
            extensions=[".parquet"],
            type_check=lambda obj: isinstance(obj, pd.DataFrame),
            saver=lambda obj, path: obj.to_parquet(path),
            loader=lambda path: pd.read_parquet(path),
            required_package="pyarrow",
        )
    """
```

### 2. Updated Load API

```python
def load_artifact(
    filename: str,
    loader: Callable[[Path], Any] | None = None,
    format: str | None = None,
) -> Any | None:
    """Load artifact with optional explicit format.

    Args:
        filename: Name of artifact to load
        loader: Optional custom loader function (path) -> object
        format: Optional format name (overrides auto-detection)

    Returns:
        Loaded object, or None if artifact doesn't exist

    Examples:
        # Auto-detect from extension (existing behavior)
        metrics = yanex.load_artifact("metrics.json")

        # Explicit format for ambiguous extensions
        workload = yanex.load_artifact("data.jsonl", format="workload")

        # Custom loader (existing behavior)
        data = yanex.load_artifact("data.bin", loader=my_loader)
    """
```

### 3. Save API (unchanged, but enhanced)

```python
def save_artifact(obj: Any, filename: str, saver: Any | None = None) -> None:
    """Save Python object to experiment artifacts.

    Now checks registered custom handlers before falling back to built-in formats.

    Args:
        obj: Python object to save
        filename: Name for saved artifact (extension determines format)
        saver: Optional custom saver function (obj, path) -> None

    Examples:
        # Auto-detect using registered handler
        yanex.save_artifact(workload, "data.jsonl")  # Uses Workload handler

        # Auto-detect built-in format
        yanex.save_artifact({"acc": 0.95}, "metrics.json")

        # Custom saver (still supported)
        yanex.save_artifact(obj, "data.bin", saver=my_saver)
    """
```

## Implementation Details

### 1. Update FormatHandler dataclass

```python
@dataclass
class FormatHandler:
    name: str  # NEW - format identifier for explicit loading
    extensions: list[str]
    type_check: Callable[[Any], bool]
    saver: Callable[[Any, Path], None]
    loader: Callable[[Path], Any]
    required_package: str | None = None
```

### 2. Update built-in handlers

```python
FORMAT_HANDLERS = [
    FormatHandler(
        name="text",  # Add names to all built-in handlers
        extensions=[".txt"],
        type_check=lambda obj: isinstance(obj, str),
        saver=_save_text,
        loader=_load_text,
    ),
    FormatHandler(
        name="jsonl",
        extensions=[".jsonl"],
        type_check=lambda obj: isinstance(obj, list),
        saver=_save_jsonl,
        loader=_load_jsonl,
    ),
    # ... update all handlers
]
```

### 3. Implement registration function

```python
# In yanex/core/artifact_formats.py

def register_format(
    name: str,
    extensions: list[str],
    type_check: Callable[[Any], bool],
    saver: Callable[[Any, Path], None],
    loader: Callable[[Path], Any],
    required_package: str | None = None,
) -> None:
    """Register custom format handler."""
    # Validate name is unique
    if any(h.name == name for h in FORMAT_HANDLERS):
        raise ValueError(f"Format '{name}' is already registered")

    # Validate extensions
    if not extensions:
        raise ValueError("At least one extension must be provided")

    # Add to registry (prepend for priority)
    handler = FormatHandler(
        name=name,
        extensions=extensions,
        type_check=type_check,
        saver=saver,
        loader=loader,
        required_package=required_package,
    )
    FORMAT_HANDLERS.insert(0, handler)
```

### 4. Update handler lookup for loading

```python
def get_handler_for_load(filename: str, format: str | None = None) -> FormatHandler:
    """Find handler for loading.

    Lookup order:
    1. If format specified, find by name
    2. Otherwise, find by extension (existing behavior)

    Args:
        filename: Filename to load
        format: Optional format name for explicit lookup

    Returns:
        FormatHandler for the file

    Raises:
        ValueError: If format not found or extension not supported
    """
    if format:
        # Explicit format - find by name
        for handler in FORMAT_HANDLERS:
            if handler.name == format:
                # Check if required package is available
                if handler.required_package:
                    try:
                        __import__(handler.required_package)
                    except ImportError as err:
                        raise ImportError(
                            f"Loading with format '{format}' requires {handler.required_package}. "
                            f"Install with: pip install {handler.required_package}"
                        ) from err
                return handler

        raise ValueError(
            f"Unknown format: '{format}'. "
            f"Available formats: {', '.join(h.name for h in FORMAT_HANDLERS)}"
        )

    # Auto-detect from extension (existing logic)
    ext = Path(filename).suffix.lower()
    for handler in FORMAT_HANDLERS:
        if ext in handler.extensions:
            if handler.required_package:
                try:
                    __import__(handler.required_package)
                except ImportError as err:
                    raise ImportError(
                        f"Loading {filename} requires {handler.required_package}. "
                        f"Install with: pip install {handler.required_package}"
                    ) from err
            return handler

    # No handler found
    supported = sorted({ext for h in FORMAT_HANDLERS for ext in h.extensions})
    raise ValueError(
        f"Cannot auto-detect format for '{filename}'. "
        f"Supported extensions: {', '.join(supported)}. "
        f"Use 'loader' parameter for custom formats."
    )
```

### 5. Update API functions

```python
# In yanex/api.py

def save_artifact(obj: Any, filename: str, saver: Any | None = None) -> None:
    """Save artifact (implementation unchanged, but gets custom handlers automatically)."""
    # Existing implementation already uses get_handler_for_save()
    # which will check custom handlers first (prepended to list)
    pass

def load_artifact(
    filename: str,
    loader: Any | None = None,
    format: str | None = None  # NEW parameter
) -> Any | None:
    """Load artifact with optional explicit format."""
    experiment_id = _get_current_experiment_id()

    if experiment_id is None:
        # Standalone mode
        from .core.artifact_io import load_artifact_from_path
        artifacts_dir = _get_standalone_artifacts_dir()
        artifact_path = artifacts_dir / filename

        if not artifact_path.exists():
            return None

        return load_artifact_from_path(artifact_path, loader, format=format)
    else:
        # Experiment mode
        manager = _get_experiment_manager()
        return manager.storage.load_artifact(
            experiment_id, filename, loader, format=format
        )
```

### 6. Export from public API

```python
# In yanex/__init__.py

from .core.artifact_formats import register_format

__all__ = [
    # ... existing exports
    "register_format",  # NEW
]
```

## Usage Examples

### Example 1: Workload objects

```python
from your_package import Workload
import yanex

# Register once (e.g., in __init__.py or config)
yanex.register_format(
    name="workload",
    extensions=[".jsonl"],
    type_check=lambda obj: isinstance(obj, Workload),
    saver=lambda obj, path: obj.save(str(path)),
    loader=lambda path: Workload.load(str(path)),
)

# Use everywhere
train = Workload(...)
yanex.save_artifact(train, "workload_train.jsonl")  # Auto-detects Workload

# Load explicitly
train = yanex.load_artifact("workload_train.jsonl", format="workload")
```

### Example 2: Cache objects

```python
import yanex

yanex.register_format(
    name="cache",
    extensions=[".cache"],
    type_check=lambda obj: isinstance(obj, Cache),
    saver=lambda obj, path: obj.serialize(path),
    loader=lambda path: Cache.deserialize(path),
)

yanex.save_artifact(cache, "embeddings.cache")
cache = yanex.load_artifact("embeddings.cache", format="cache")
```

### Example 3: Third-party library support

```python
# Library author can provide registration helper
import yanex

def register_mylib_formats():
    """Register mylib custom formats with yanex."""
    yanex.register_format(
        name="mylib_model",
        extensions=[".mylib"],
        type_check=lambda obj: isinstance(obj, mylib.Model),
        saver=lambda obj, path: obj.save_to_file(path),
        loader=lambda path: mylib.Model.load_from_file(path),
    )

# Users just call once
import mylib.yanex_integration
mylib.yanex_integration.register_mylib_formats()
```

## Handler Priority and Conflicts

### Save (type-based)
- Custom handlers checked first (prepended to list)
- Built-in handlers checked second
- First matching type wins
- If multiple custom handlers match same type, first registered wins

### Load (extension-based)
- Without `format=`: First handler for extension wins (custom handlers first)
- With `format=`: Explicit handler used, ignores extension ambiguity
- If extension has multiple handlers, user must specify `format=`

### Conflict resolution
```python
# Built-in: list -> jsonl
yanex.save_artifact([1, 2, 3], "data.jsonl")  # Uses built-in jsonl handler

# Custom: Workload -> jsonl
yanex.save_artifact(workload, "data.jsonl")  # Uses custom workload handler

# Loading ambiguous extension
data = yanex.load_artifact("data.jsonl")  # Uses first handler (custom if registered)
data = yanex.load_artifact("data.jsonl", format="jsonl")  # Explicit built-in
data = yanex.load_artifact("data.jsonl", format="workload")  # Explicit custom
```

## Migration Path

### Phase 1: Add infrastructure (v0.7.0)
- Add `name` field to FormatHandler
- Update all built-in handlers with names
- Add `format` parameter to load_artifact
- Update handler lookup logic

### Phase 2: Add registration API (v0.7.0)
- Implement register_format()
- Export from yanex.__init__
- Add tests for custom handlers
- Document in API reference

### Phase 3: Deprecation (future, if needed)
- No deprecations needed - fully backwards compatible
- Custom `saver`/`loader` parameters still supported

## Testing Strategy

### Unit tests
- Test registration with valid/invalid inputs
- Test handler priority (custom before built-in)
- Test format parameter in load_artifact
- Test error handling (unknown format, missing package)
- Test extension conflicts

### Integration tests
- Test save/load roundtrip with custom handler
- Test multiple handlers for same extension
- Test standalone vs experiment mode
- Test with dependencies (load from parent experiment)

### Example tests
```python
def test_custom_handler_registration():
    """Test registering custom format handler."""
    class CustomObj:
        def __init__(self, data):
            self.data = data

        def save(self, path):
            Path(path).write_text(str(self.data))

        @classmethod
        def load(cls, path):
            data = Path(path).read_text()
            return cls(data)

    # Register handler
    yanex.register_format(
        name="custom",
        extensions=[".custom"],
        type_check=lambda obj: isinstance(obj, CustomObj),
        saver=lambda obj, path: obj.save(path),
        loader=lambda path: CustomObj.load(path),
    )

    # Test save/load
    obj = CustomObj("test data")
    yanex.save_artifact(obj, "data.custom")
    loaded = yanex.load_artifact("data.custom", format="custom")
    assert loaded.data == "test data"

def test_format_parameter_with_shared_extension():
    """Test explicit format with ambiguous extension."""
    # Register custom jsonl handler
    yanex.register_format(
        name="custom_jsonl",
        extensions=[".jsonl"],
        type_check=lambda obj: isinstance(obj, dict) and "custom" in obj,
        saver=lambda obj, path: ...,
        loader=lambda path: {"custom": True, "loaded": True},
    )

    # Save with custom handler
    yanex.save_artifact({"custom": True}, "data.jsonl")

    # Load with explicit format
    data = yanex.load_artifact("data.jsonl", format="custom_jsonl")
    assert data["custom"] == True
    assert data["loaded"] == True
```

## Documentation Updates

### 1. New documentation page: `docs/custom-formats.md`
- Overview of format registration
- Common use cases
- API reference for register_format()
- Examples for different object types
- Best practices

### 2. Update existing docs
- `docs/artifacts.md`: Add section on custom formats
- `docs/run-api.md`: Document format parameter in load_artifact
- `docs/README.md`: Link to custom formats guide

### 3. API docstrings
- Update load_artifact() docstring with format examples
- Add comprehensive docstring to register_format()

## Open Questions

1. **Should we allow unregistering formats?**
   - Use case: Testing, cleanup
   - API: `yanex.unregister_format(name)`
   - Decision: Add in future if needed

2. **Should we support format aliases?**
   - Use case: "yml" and "yaml" both work
   - API: `FormatHandler.aliases: list[str]`
   - Decision: Not needed initially, extensions handle this

3. **Should we validate handler uniqueness?**
   - Currently: First matching handler wins
   - Alternative: Raise error if multiple handlers match
   - Decision: Current approach is more flexible

4. **Should format registration be thread-safe?**
   - Use case: Multi-threaded applications
   - Solution: Add lock around FORMAT_HANDLERS modification
   - Decision: Add if users report issues

5. **Should we persist registered formats?**
   - Use case: Reproducibility across runs
   - Solution: Save registered formats in experiment metadata
   - Decision: Not needed - registration is code-level configuration

## Files to Modify

1. `yanex/core/artifact_formats.py`
   - Add `name` field to FormatHandler
   - Update all built-in handlers
   - Implement register_format()
   - Update get_handler_for_load() with format parameter

2. `yanex/core/artifact_io.py`
   - Update load_artifact_from_path() signature with format parameter
   - Pass format to get_handler_for_load()

3. `yanex/core/storage_artifacts.py`
   - Update load_artifact() signature with format parameter
   - Pass format through to artifact_io

4. `yanex/api.py`
   - Update load_artifact() signature with format parameter
   - Pass format to storage layer

5. `yanex/__init__.py`
   - Export register_format

6. `tests/core/test_artifact_formats.py`
   - Add tests for custom handler registration
   - Add tests for format parameter
   - Add tests for handler priority

7. `docs/custom-formats.md` (NEW)
   - Complete guide for custom formats

8. `docs/run-api.md`
   - Update load_artifact documentation

## Success Criteria

1. Users can register custom formats with simple API call
2. Type-based auto-detection works for saving
3. Format parameter works for loading ambiguous extensions
4. No breaking changes to existing code
5. Comprehensive tests with 90%+ coverage
6. Clear documentation with examples
7. Third-party libraries can provide yanex integrations

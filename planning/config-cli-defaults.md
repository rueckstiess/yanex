# Implementation Plan: Config-Based CLI Defaults for Run Command

## Overview

Add support for setting CLI parameters of the `yanex run` command via the config file under a special `yanex` key. This allows users to set default values for CLI options like `--name`, `--tag`, `--description`, `--ignore-dirty`, etc. in their config file while maintaining CLI argument override precedence.

## Requirements

1. **Scope**: Only for the `yanex run` command (not other commands)
2. **Naming**: Use CLI parameter names with dash-to-underscore conversion (`--ignore-dirty` → `ignore_dirty`)
3. **Precedence**: CLI args > config defaults > built-in defaults
4. **Backwards Compatibility**: Existing configs without `yanex` key must work unchanged
5. **Integration**: Reuse existing `resolve_config()` function rather than adding separate config parsing

## Current Architecture

### Current Flow
1. **CLI Parsing**: Click collects CLI arguments into variables (`name`, `tag`, `description`, etc.)
2. **Config Loading**: `load_and_merge_config()` processes config file + `--param` overrides → `merged_config`
3. **Separate Processing**: CLI args and config passed separately to experiment functions

### Key Files
- `yanex/cli/commands/run.py:50-62` - CLI option definitions
- `yanex/core/config.py:186-222` - `resolve_config()` function
- `yanex/cli/_utils.py:13-48` - `load_and_merge_config()` wrapper

### Current CLI Options (Run Command)
| CLI Option | Type | Description |
|------------|------|-------------|
| `--name/-n` | `str` | Experiment name |
| `--tag/-t` | `list[str]` (multiple) | Experiment tags |
| `--description/-d` | `str` | Experiment description |
| `--dry-run` | `bool` (flag) | Validate without running |
| `--ignore-dirty` | `bool` (flag) | Allow uncommitted changes |
| `--stage` | `bool` (flag) | Stage for later execution |

## Implementation Plan

### 1. Config File Format

```yaml
# CLI defaults under 'yanex' key
yanex:
  name: my-experiment
  tag: [dev, testing]  # Multiple tags allowed, or single: tag: dev
  description: "Default experiment description"
  ignore_dirty: true
  dry_run: false
  stage: false

# Regular experiment parameters (unchanged)
learning_rate: 0.004
workload:
  query_num: 10000
```

### 2. Core Changes

#### A. Extend `resolve_config()` Function

**File**: `yanex/core/config.py:186-222`

**Change**: Modify return type to include CLI defaults:

```python
def resolve_config(
    config_path: Path | None = None,
    param_overrides: list[str] | None = None,
    default_config_name: str = "config.yaml",
) -> tuple[dict[str, Any], dict[str, Any]]:  # NEW: return tuple
    """
    Resolve final configuration from file and parameter overrides.
    
    Returns:
        Tuple of (experiment_config, cli_defaults)
    """
    # Load config file (existing logic)
    config = {}
    if config_path is None:
        default_path = Path.cwd() / default_config_name
        if default_path.exists():
            config_path = default_path
    
    if config_path is not None:
        config = load_yaml_config(config_path)
    
    # NEW: Extract CLI defaults from 'yanex' key
    cli_defaults = config.pop('yanex', {})  # Remove from main config
    
    # Apply parameter overrides to experiment config only (existing logic)
    if param_overrides:
        override_config = parse_param_overrides(param_overrides)
        config = merge_configs(config, override_config)
    
    return config, cli_defaults
```

#### B. Update CLI Utils

**File**: `yanex/cli/_utils.py:13-48`

**Change**: Update `load_and_merge_config()` to handle new return format:

```python
def load_and_merge_config(
    config_path: Path | None, param_overrides: list[str], verbose: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:  # NEW: return tuple
    """
    Load and merge configuration from various sources.
    
    Returns:
        Tuple of (experiment_config, cli_defaults)
    """
    try:
        experiment_config, cli_defaults = resolve_config(
            config_path=config_path,
            param_overrides=param_overrides,
        )
        
        if verbose:
            if config_path:
                click.echo(f"Loaded config from: {config_path}")
            if cli_defaults:
                click.echo(f"Loaded CLI defaults: {cli_defaults}")
        
        return experiment_config, cli_defaults
        
    except Exception as e:
        raise click.ClickException(f"Failed to load configuration: {e}") from e
```

#### C. Update Run Command

**File**: `yanex/cli/commands/run.py:134-185`

**Changes**:

1. **Update config loading** (around line 136):
```python
# Load and merge configuration
experiment_config, cli_defaults = load_and_merge_config(
    config_path=config, param_overrides=list(param), verbose=verbose
)
```

2. **Add CLI resolution logic** (after line 138):
```python
# Resolve CLI parameters with config defaults and CLI overrides
resolved_name = name if name is not None else cli_defaults.get('name')
resolved_tags = list(tag) if tag else _normalize_tags(cli_defaults.get('tag', []))
resolved_description = description if description is not None else cli_defaults.get('description')
resolved_ignore_dirty = ignore_dirty or cli_defaults.get('ignore_dirty', False)
resolved_dry_run = dry_run or cli_defaults.get('dry_run', False)
resolved_stage = stage or cli_defaults.get('stage', False)
```

3. **Add helper function** (at end of file):
```python
def _normalize_tags(tag_value: Any) -> list[str]:
    """Convert config tag value to list format matching CLI --tag behavior."""
    if isinstance(tag_value, str):
        return [tag_value]
    elif isinstance(tag_value, list):
        return [str(t) for t in tag_value]
    else:
        return []
```

4. **Update function calls** (lines 144-185):
   - Replace `name` with `resolved_name`
   - Replace `list(tag)` with `resolved_tags`
   - Replace `description` with `resolved_description`
   - Replace `ignore_dirty` with `resolved_ignore_dirty`
   - Update `merged_config` to `experiment_config`

### 3. Testing Strategy

#### Unit Tests
- **Config parsing**: Test `yanex` section extraction and validation
- **CLI resolution**: Test precedence rules (CLI > config > defaults)
- **Tag normalization**: Test string vs list handling

#### Integration Tests
- **End-to-end**: Test complete run command with config-based defaults
- **Override behavior**: Verify CLI args override config defaults
- **Backwards compatibility**: Ensure existing configs work unchanged

#### Test Files to Update
- `tests/core/test_config.py` - Add tests for new `resolve_config()` behavior
- `tests/cli/test_main.py` - Add integration tests for run command

### 4. Migration Path

1. **Phase 1**: Implement core functionality (config parsing, CLI resolution)
2. **Phase 2**: Add comprehensive tests
3. **Phase 3**: Update documentation and examples

### 5. Example Usage

**Before** (CLI only):
```bash
yanex run train.py --name "lr-experiment" --tag dev --tag testing --ignore-dirty
```

**After** (config + CLI):
```yaml
# config.yaml
yanex:
  name: lr-experiment
  tag: [dev, testing]
  ignore_dirty: true

learning_rate: 0.01
```

```bash
yanex run train.py  # Uses config defaults
yanex run train.py --name "override-name"  # CLI overrides config name
```

## Implementation Notes

- **Naming Convention**: CLI dashes become underscores (`--ignore-dirty` → `ignore_dirty`)
- **Type Consistency**: Tags can be single string or list in config, normalized to list
- **Validation**: Existing validation logic applies to resolved values
- **Error Handling**: Invalid `yanex` section values should produce clear error messages
- **Future Extensions**: This pattern could be extended to other commands later if needed

## Backwards Compatibility

- Existing configs without `yanex` key work unchanged
- Existing CLI usage patterns work unchanged  
- Only additive changes to config format
- No breaking changes to function signatures (return type changes are compatible)
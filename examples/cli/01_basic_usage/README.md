# 01: Basic Usage

## What This Example Demonstrates

- Core yanex API: `get_param()` and `log_metrics()`
- Scripts that work both standalone and with yanex tracking
- Parameter defaults
- Simple metric logging

## Files

- `compute.py` - Minimal example showing parameter access and metric logging

## How to Run

### Run without yanex (standalone)
```bash
python compute.py
```

### Run with yanex tracking
```bash
yanex run compute.py
```

### Override parameters with `--param` (or `-p`)
```bash
yanex run compute.py --param num_items=5000
yanex run compute.py -p num_items=2000 -p delay_ms=50
```

### Add metadata
```bash
yanex run compute.py --name "quick-test" --tag benchmark
```

## Expected Output

```
Processing 1000 items with 100ms delay...
Completed in 0.100 seconds
Throughput: 9995 items/second
```

With yanex, you'll also see:
```
✓ Experiment completed successfully: abc12345
  Directory: /Users/you/.yanex/experiments/abc12345
```

## What to Look For

After running with yanex:
- **List experiments**: `yanex list`
- **Show experiment details**: `yanex show <id>`
- **Open experiment directory**: `yanex open <id>`
- **Check experiment directory**: `~/.yanex/experiments/<id>/`
  - `metadata.json` - Experiment status, timing, git state
  - `config.json` - Parameters used (num_items, delay_ms)
  - `metrics.json` - Logged metrics (total_time_seconds, items_per_second, num_items)

## Key Concepts

- **Dual-mode scripts**: Works with or without yanex
- **Parameter defaults**: Uses defaults when not specified
- **No-op in standalone**: `log_metrics()` does nothing without yanex
- **Automatic tracking**: Git state, timing, environment captured automatically

## Next Steps

- Try different parameter values and compare results: `yanex compare`
- See example 02 to learn about configuration files

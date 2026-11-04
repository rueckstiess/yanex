# 11: Metadata Templates

## What This Example Demonstrates

- Parameter placeholders in names, tags, and descriptions
- Sweep names without automatically appended suffixes
- Python number formatting and YAML metadata defaults
- Local-first parameter lookup through transitive upstream dependencies
- Agreeing values versus conflicting values in multiple dependencies
- Previewing metadata before any experiments run

## Files

- `benchmark.py` - Tiny batch-processing benchmark; works standalone or tracked
- `prepare_data.py` - Creates a small data artifact for dependency examples
- `template-config.yaml` - Parameter sweep with metadata templates

The workload is deliberately small and uses only Python and Yanex. Timings are
illustrative, not a meaningful performance comparison.

## How to Run

Run these commands from this example's directory:

```bash
cd examples/cli/11_metadata_templates
```

When testing the feature from a source checkout, prefix the commands below with
`uv run` (for example, `uv run yanex run benchmark.py`). This selects the checkout's
Yanex installation. The examples use a `metadata-templates` tag for easy filtering.

### 1. Standalone and single-run templates

```bash
python benchmark.py

yanex run benchmark.py -p batch_size=50 \
  -n "benchmark-bs-{batch_size}" \
  -t metadata-templates -t "batch-{batch_size}" \
  -d "Process the built-in dataset in batches of {batch_size}"
```

The tracked run is named `benchmark-bs-50` and has a `batch-50` tag. With 1,000
items, the script processes 20 batches and reports a checksum of `499500.00`.

Parameters referenced by templates must exist in the saved configuration or an
upstream experiment. A Python default passed to `get_param()` is only applied
when the script runs, so it is **not available to template preflight**. This is
why the command explicitly supplies `-p batch_size=50`.

### 2. Preview and run a sweep

```bash
yanex run benchmark.py -p batch_size=10,50,100,200 \
  -n "benchmark-bs-{batch_size}" \
  -t metadata-templates -t "batch-{batch_size}" \
  -d "Batch size: {batch_size}" --dry-run
```

The preview shows `benchmark-bs-10`, `benchmark-bs-50`, `benchmark-bs-100`, and
`benchmark-bs-200`. Nothing is executed or staged. Remove `--dry-run` to run the
sweep, or replace it with `--parallel 2` to run two experiments at a time.

A name containing a placeholder is used exactly as rendered: Yanex does not add
another batch-size suffix. Plain names retain the existing automatic sweep naming.

### 3. Formatting and YAML defaults

```bash
yanex run benchmark.py -c template-config.yaml --dry-run

# Run the same four combinations
yanex run benchmark.py -c template-config.yaml --parallel 2
```

The first name is `batch-010-scale-0.12`:

- `{batch_size:03d}` pads an integer to three digits.
- `{multiplier:.2f}` displays two decimal places.
- `{multiplier:.3g}` displays three significant digits in the description.

The saved multiplier remains `0.125`; formatting changes only the text. Nested
paths such as `{model.learning_rate:.3g}` are also supported, provided that
parameter exists. An unambiguous suffix such as `{learning_rate:.3g}` works too.

### 4. Upstream and transitive lookup

First prepare data, explicitly recording its label and size:

```bash
yanex run prepare_data.py -p dataset=toy -p num_items=1000 \
  -n "source-{dataset}" -t metadata-templates
```

Copy the completed experiment ID printed by Yanex into `DATA_ID` below (replace
`PASTE_DATA_ID` with the actual ID):

```bash
DATA_ID=PASTE_DATA_ID

yanex run benchmark.py -D data="$DATA_ID" -p batch_size=50 \
  -n "{dataset}-bs-{batch_size}" -t metadata-templates \
  -d "Process {num_items} items from {dataset}"
```

The name is `toy-bs-50`. Only `batch_size` is local; `dataset` and `num_items` come
from the data experiment. The script loads that experiment's `data.json` artifact.

Now copy the benchmark's ID into `BENCHMARK_ID` and run a second benchmark:

```bash
BENCHMARK_ID=PASTE_BENCHMARK_ID

yanex run benchmark.py -D previous="$BENCHMARK_ID" -p batch_size=100 \
  -n "{dataset}-bs-{batch_size}" -t metadata-templates \
  -d "Transitive lookup: {num_items} items from {dataset}"
```

The name is `toy-bs-100`. `dataset` and `num_items` are found two steps upstream,
and the data artifact is loaded transitively. Local `batch_size=100` wins over
the previous benchmark's `batch_size=50`. Template lookup does not copy upstream
parameters into the local configuration or change the behavior of `get_param()`.

### 5. Agreement and conflicts

Create another source with the same recorded values:

```bash
yanex run prepare_data.py -p dataset=toy -p num_items=1000 \
  -n "second-source-{dataset}" -t metadata-templates
```

Copy its ID into `SECOND_ID`. Previewing a name with both sources succeeds
because both define `dataset=toy`:

```bash
SECOND_ID=PASTE_SECOND_ID

yanex run benchmark.py -D first="$DATA_ID" -D second="$SECOND_ID" \
  -n "compare-{dataset}" -t metadata-templates --dry-run
```

Keep this as a dry run: both sources contain `data.json`, so executing this
particular benchmark would require an explicit artifact-source choice.

To see a conflict, create a source with a different label, copy its ID into
`OTHER_ID`, then preview both sources:

```bash
yanex run prepare_data.py -p dataset=other -p num_items=1000 \
  -n "source-{dataset}" -t metadata-templates

OTHER_ID=PASTE_OTHER_ID

yanex run benchmark.py -D first="$DATA_ID" -D second="$OTHER_ID" \
  -n "compare-{dataset}" -t metadata-templates --dry-run
```

The final command intentionally fails and identifies the conflicting experiments
and values. No new experiment is created. Adding `-p dataset=chosen` would resolve
the **metadata** conflict through local precedence; it would not select an artifact.

## Useful Rules

- Missing parameters and ambiguous paths fail before any sweep member starts.
- Repeated values at the same parameter path are allowed when they agree.
- Names and tags reject glob characters (`*`, `?`, `[` and `]`). Tags also reject
  whitespace and commas.
- Use `{{` and `}}` for literal braces, for example `-d "Value: {{batch_size}}"`.
- Python expressions, indexing, and function calls are not supported.
- `--stage` saves rendered metadata; later execution does not render it again.

## Inspect Results

```bash
yanex list -t metadata-templates
```

See [04: Metadata and Tags](../04_metadata_and_tags/README.md),
[07: Parameter Sweeps](../07_parameter_sweeps/README.md), and
[10: Dependencies](../10_dependencies/README.md) for the underlying features.

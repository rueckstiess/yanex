# Experiment Logging Format

## Log Location

**Default**: `scripts/experiment-log.md`

If this location doesn't exist or seems wrong, confirm with the user:
> "I'll log experiments to `scripts/experiment-log.md`. Is that the right location?"

Once confirmed, don't ask again in the same session.

## Log Structure

Each experiment group gets:
1. **Header** with group name and date
2. **Summary table** with all experiments
3. **Context note** explaining what was run
4. **Command block** with the exact `yanex run` invocation

## Template

```markdown
## {group-prefix} - {description}
_{date}_

| ID | Name | Sweep | Dependencies |
|----|------|-------|--------------|
| `{id1}` | {name1} | {sweep-value} | {deps} |
| `{id2}` | {name2} | {sweep-value} | {deps} |

{user's context/description of what they're doing}:
```bash
yanex run {script} {args}
```

---
```

## Field Formats

### ID Column
- Use backticks: `` `abc12345` ``
- Always 8 characters

### Name Column
- The experiment name as specified with `-n`

### Sweep Column
- Show the **concrete value** for that specific experiment row
- Examples: `lr=0.001`, `epochs=20`, `batch_size=64`
- Use `-` if no sweep parameters

### Dependencies Column
- Format: `slot=`id``
- Examples:
  - Single: `data=`abc12345``
  - Multiple: `data=`abc123`, model=`def456``
- Use `-` if no dependencies

## Example Log

```markdown
# Experiment Log

## yelp-3 - Lower Learning Rate Sweep
_2025-11-30_

| ID | Name | Sweep | Dependencies |
|----|------|-------|--------------|
| `a1b2c3d4` | yelp-3-train-lr | lr=1e-3 | data=`12345678` |
| `e5f6g7h8` | yelp-3-train-lr | lr=1e-4 | data=`12345678` |

Running roberta with lower learning rate:
```bash
yanex run scripts/02_train_encoder.py -c config-yelp.yaml -D data=12345678 -p train.lr=1e-3,1e-4 -n yelp-3-train-lr
```

---

## yelp-2 - Baseline Training
_2025-11-29_

| ID | Name | Sweep | Dependencies |
|----|------|-------|--------------|
| `11112222` | yelp-2-train-baseline | - | data=`abcd1234` |

Initial baseline with default params:
```bash
yanex run scripts/02_train_encoder.py -c config-yelp.yaml -D data=abcd1234 -n yelp-2-train-baseline
```

---

## yelp-1 - Data Preparation
_2025-11-28_

| ID | Name | Sweep | Dependencies |
|----|------|-------|--------------|
| `abcd1234` | yelp-1-data-10k | - | - |

Preparing 10k sample dataset:
```bash
yanex run scripts/01_prepare_data.py -c config-yelp.yaml -p samples=10000 -n yelp-1-data-10k
```

---
```

## Handling Failures

**Do NOT log failed experiments to the markdown file.**

Instead, alert the user in conversation:
> "Note: Experiment `xyz789` failed. You can inspect it with `yanex show xyz789`."

## Multi-Sweep Experiments

When a sweep creates multiple experiments, list each one as a separate row:

```markdown
| ID | Name | Sweep | Dependencies |
|----|------|-------|--------------|
| `aaa11111` | yelp-3-hpo | lr=0.001, bs=32 | data=`abc123` |
| `bbb22222` | yelp-3-hpo | lr=0.001, bs=64 | data=`abc123` |
| `ccc33333` | yelp-3-hpo | lr=0.01, bs=32 | data=`abc123` |
| `ddd44444` | yelp-3-hpo | lr=0.01, bs=64 | data=`abc123` |
```

## Appending to Existing Log

When adding to an existing log:
1. Read the current file
2. Add new content at the **top** (after the `# Experiment Log` header)
3. Keep the `---` separator between groups

# 04: Metadata and Tags

## What This Example Demonstrates

- Naming experiments with `--name`
- Adding descriptions with `--description`
- Tagging experiments with `--tag` (or `-t`, can be used multiple times)
- Setting metadata defaults in config files (under `yanex:` section)
- Filtering experiments by tags and names
- Cleaning up experiments by tags

## Files

- `scrape.py` - Minimal web scraping simulation
- `scraper-config.yaml` - Config with `yanex:` metadata section

## Why Metadata Matters

As you run more experiments, organization becomes critical. Metadata helps you:
- **Find experiments quickly** - Search by name patterns or tags
- **Understand experiments** - Descriptions remind you what each experiment tested
- **Group related work** - Tags organize experiments by project, feature, or environment
- **Clean up efficiently** - Delete old experiments by tag

## How to Run

### Basic usage (no metadata)
```bash
yanex run scrape.py
```

### Add a name
```bash
yanex run scrape.py --name "baseline-scraper"
```

### Add description
```bash
yanex run scrape.py \
  --name "fast-scraper" \
  --description "Testing with reduced delays"
```

### Add tags (multiple tags allowed)
```bash
# Single tag
yanex run scrape.py --tag production

# Multiple tags (using -t shortcut)
yanex run scrape.py -t scraping -t production -t v2
```

### Combine metadata with parameters
```bash
yanex run scrape.py \
  --name "large-scale-test" \
  --description "Testing with 100 pages" \
  --tag scraping --tag performance \
  -p num_pages=100
```

### Use config file with metadata defaults
```bash
# Uses metadata from scraper-config.yaml's yanex: section
yanex run scrape.py --config scraper-config.yaml

# Override metadata from config
yanex run scrape.py --config scraper-config.yaml \
  --name "custom-name" \
  -t production  # Adds to config tags
```

## Config File Metadata

Set default metadata in your config file under a `yanex:` section:

```yaml
# scraper-config.yaml
num_pages: 10
delay_ms: 100

yanex:
  name: "web-scraper-test"
  description: "Testing web scraping with different parameters"
  tag:
    - scraping
    - automation
    - dev
```

**Behavior:**
- CLI `--name` **replaces** config name
- CLI `--description` **replaces** config description
- CLI `--tag` **adds to** config tags (they merge!)

## Finding Experiments

### List by tags
```bash
# Experiments with a specific tag
yanex list --tag production
yanex list -t scraping

# Multiple tags (AND logic - must have ALL tags)
yanex list -t scraping -t production
```

### List by name pattern
```bash
# Glob pattern matching
yanex list --name "baseline-*"
yanex list --name "*scraper*"
```

### Combine filters
```bash
# Production scraping experiments
yanex list --tag production --tag scraping

# Completed production experiments
yanex list --status completed --tag production
```

### Show all metadata
```bash
# Get the experiment ID from yanex list
yanex show <experiment_id>
```

## Cleaning Up

### Delete by tags
```bash
# Delete all dev experiments
yanex delete --tag dev

# Delete specific combination
yanex delete --tag scraping --tag testing
```

### Delete by script name
```bash
# Delete all experiments from a specific script
yanex delete --script scrape.py
yanex delete -c scrape.py  # Short flag

# Combine with other filters
yanex delete --script scrape.py --tag testing
```

### Delete by name pattern
```bash
# Delete all baseline experiments
yanex delete --name "baseline-*"
```

### Delete by status
```bash
# Clean up failed experiments
yanex delete --status failed

# Remove old completed tests
yanex delete --status completed --tag testing
```

## Expected Output

```
Scraping 10 pages with 100ms delay...
  Processed 5/10 pages...
  Processed 10/10 pages...
Scraping complete! Found 127 items total.
✓ Experiment completed successfully: abc12345
```

## Metadata Best Practices

### Naming Conventions
- **Descriptive names**: `"baseline-v1"`, `"optimized-scraper"`
- **Include version**: `"model-v2"`, `"api-test-v3"`
- **Be consistent**: Use same pattern across related experiments

### Tag Strategies
- **Environment**: `dev`, `staging`, `production`
- **Project/Feature**: `user-auth`, `search-engine`, `api-v2`
- **Experiment type**: `baseline`, `ablation`, `optimization`
- **Status/Phase**: `testing`, `validated`, `archived`

### Description Guidelines
- **What you tested**: "Testing pagination with 100 pages"
- **Why you ran it**: "Investigating timeout issues"
- **What changed**: "Increased delay from 50ms to 100ms"

## Example Workflow

```bash
# 1. Run baseline
yanex run scrape.py --name "baseline" -t scraping -t baseline

# 2. Run optimization test
yanex run scrape.py -p delay_ms=50 \
  --name "fast-scraper" \
  --description "Testing with reduced delay" \
  -t scraping -t optimization

# 3. Run production test
yanex run scrape.py -p num_pages=100 \
  --name "production-test" \
  -t scraping -t production

# 4. Compare results
yanex compare --tag scraping

# 5. Clean up baseline experiments
yanex delete --tag baseline
```

## What to Look For

- **View tagged experiments**: `yanex list --tag scraping`
- **Find by name**: `yanex list --name "*scraper*"`
- **Check metadata**: `yanex show <id>` displays name, description, tags
- **Filter combinations**: Multiple filters work together

## Key Concepts

- **Tags enable organization**: Group experiments by project, environment, purpose
- **Names aid discovery**: Descriptive names make experiments easy to find
- **Descriptions provide context**: Future you will thank present you
- **Config defaults save time**: Set common metadata in config files
- **Filtering is powerful**: Combine status, tags, names, and dates

## Next Steps

- Organize your experiments with consistent tagging
- Use `yanex compare --tag <tag>` to compare related experiments
- See example 05 to learn about parameter sweeps

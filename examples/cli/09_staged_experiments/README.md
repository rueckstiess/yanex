# 09: Staged Experiments

## What This Example Demonstrates

- Creating experiments without running them using `--stage`
- Listing staged experiments with `yanex list --status staged`
- Running staged experiments with `yanex run --staged`
- Running staged experiments in parallel with `--parallel` / `-j`
- Canceling staged experiments
- Use cases: batch processing, overnight runs, remote execution

## Files

- `monitor_system.py` - System monitoring simulation script

## Why Stage Experiments?

Staging lets you **prepare** experiments now and **execute** them later:

- **Batch Processing**: Set up 100 experiments, run them overnight
- **Resource Planning**: Stage experiments when cluster is busy, run when free
- **Remote Execution**: Prepare on laptop, execute on server
- **Review Before Running**: Check experiment queue before starting
- **Parallel Execution**: Run many staged experiments simultaneously

## How to Run

### Stage a single experiment
```bash
# Create experiment but don't run it yet
yanex run monitor_system.py --stage
```

### Stage with parameters
```bash
# Stage with specific parameters
yanex run monitor_system.py \
  --stage \
  -p duration=10 \
  -p metric_type=cpu
```

### Stage multiple experiments (sweep)
```bash
# Stage all combinations (3 metric types)
yanex run monitor_system.py \
  --stage \
  -p "metric_type=cpu,memory,disk"

# Creates 3 staged experiments
```

### Stage a grid search
```bash
# Stage grid: 3 metrics × 3 durations = 9 experiments
yanex run monitor_system.py \
  --stage \
  -p "metric_type=cpu,memory,disk" \
  -p "duration=5,10,15"
```

### List staged experiments
```bash
# View all staged experiments
yanex list --status staged

# View with details
yanex list --status staged --verbose
```

### Run all staged experiments (sequential)
```bash
# Execute staged experiments one by one
yanex run --staged
```

### Run staged experiments in parallel
```bash
# Run with 4 parallel workers
yanex run --staged --parallel 4
yanex run --staged -j 4  # Short flag

# Auto-detect CPU count
yanex run --staged -j 0
```

### Cancel staged experiments
```bash
# Cancel specific experiment
yanex delete <experiment_id>

# Cancel all staged experiments
yanex delete --status staged
```

## Expected Output

### Staging experiments
```bash
$ yanex run monitor_system.py --stage -p "metric_type=cpu,memory,disk"
✓ Experiment staged: abc12345 (metric_type=cpu)
✓ Experiment staged: def67890 (metric_type=memory)
✓ Experiment staged: ghi11111 (metric_type=disk)

3 experiments staged. Run with: yanex run --staged
```

### Listing staged
```bash
$ yanex list --status staged
ID        Status  Created              Script             Parameters
abc12345  staged  2024-01-15 10:30:00  monitor_system.py  metric_type=cpu
def67890  staged  2024-01-15 10:30:00  monitor_system.py  metric_type=memory
ghi11111  staged  2024-01-15 10:30:00  monitor_system.py  metric_type=disk
```

### Running staged (sequential)
```bash
$ yanex run --staged
Running 3 staged experiments...

[1/3] Running abc12345 (metric_type=cpu)...
Monitoring cpu for 5 seconds (sampling every 1s)...
  Sample 1/5: cpu=45.2%
  Sample 2/5: cpu=52.1%
  ...
✓ Experiment completed: abc12345

[2/3] Running def67890 (metric_type=memory)...
Monitoring memory for 5 seconds (sampling every 1s)...
  ...
✓ Experiment completed: def67890

[3/3] Running ghi11111 (metric_type=disk)...
  ...
✓ Experiment completed: ghi11111

✓ All 3 experiments completed
```

### Running staged (parallel)
```bash
$ yanex run --staged -j 3
Running 3 staged experiments with 3 parallel workers...
✓ Completed 3/3 experiments
```

## Common Workflows

### 1. Overnight Batch Processing

```bash
# Afternoon: Stage experiments
yanex run train.py --stage \
  -p "learning_rate=0.001,0.01,0.1" \
  -p "batch_size=16,32,64" \
  --tag overnight-sweep

# Review what will run
yanex list --status staged

# Evening: Start batch run
yanex run --staged -j 4

# Morning: Check results
yanex list --status completed --tag overnight-sweep
yanex compare --tag overnight-sweep
```

### 2. Cluster Execution

```bash
# On laptop: Prepare experiments
yanex run benchmark.py --stage \
  -p "threads=1,2,4,8,16" \
  -p "workload=read,write,mixed"

# Transfer experiments directory to cluster
scp -r ~/.yanex/experiments/ cluster:~/.yanex/

# On cluster: Run experiments
yanex run --staged -j 16

# Transfer results back
scp -r cluster:~/.yanex/experiments/ ~/.yanex/
```

### 3. Review Before Execute

```bash
# Stage complex sweep
yanex run experiment.py --stage \
  -p "param1=range(1, 100)" \
  -p "param2=linspace(0.1, 1.0, 10)"

# Count experiments
yanex list --status staged | wc -l

# Review parameters
yanex list --status staged --verbose

# Decide: run or cancel
yanex run --staged -j 8  # Run
# OR
yanex delete --status staged  # Cancel
```

### 4. Incremental Staging

```bash
# Stage experiments over time
yanex run exp.py --stage -p variant=A --name "baseline-A"
yanex run exp.py --stage -p variant=B --name "baseline-B"
yanex run exp.py --stage -p variant=C --name "baseline-C"

# Later: run all at once
yanex run --staged -j 3
```

## Staged Experiment Status Flow

```
--stage
   ↓
[staged] → (yanex run --staged) → [running] → [completed/failed]
   ↓
(yanex delete) → [deleted]
```

Staged experiments remain `staged` until:
- Executed with `yanex run --staged`
- Deleted with `yanex delete <id>` or `yanex delete --status staged`

## Combining Stage with Other Features

### Stage with metadata
```bash
yanex run monitor_system.py --stage \
  -p duration=30 \
  --name "long-monitoring-run" \
  --tag monitoring --tag production \
  --description "30-second production monitoring test"
```

### Stage with config file
```bash
# Create config-cpu.yaml, config-memory.yaml, config-disk.yaml
yanex run monitor_system.py --stage --config config-cpu.yaml
yanex run monitor_system.py --stage --config config-memory.yaml
yanex run monitor_system.py --stage --config config-disk.yaml

# Run all
yanex run --staged
```

### Stage sweeps in parallel
```bash
# Stage large grid
yanex run monitor_system.py --stage \
  -p "metric_type=cpu,memory,disk,network" \
  -p "duration=5,10,15,20" \
  -p "interval=1,2,3"
# Creates 4 × 4 × 3 = 48 experiments

# Run in parallel
yanex run --staged -j 8
```

## Managing Staged Experiments

### View details of staged experiment
```bash
yanex show <experiment_id>
```

### Filter staged experiments
```bash
# By tag
yanex list --status staged --tag monitoring

# By name pattern
yanex list --status staged --name "*cpu*"

# By date
yanex list --status staged --since "1 hour ago"
```

### Cancel specific subset
```bash
# Cancel by tag
yanex delete --status staged --tag testing

# Cancel by name
yanex delete --status staged --name "test-*"
```

## Parallel Execution Details

When running staged experiments in parallel:

```bash
yanex run --staged -j 4
```

- **Process isolation**: Each experiment runs in separate process
- **Independent execution**: Experiments don't affect each other
- **Shared resources**: All write to same `~/.yanex/experiments/` directory
- **Progress tracking**: See completion status in real-time
- **Error handling**: Failed experiments don't stop others

**Worker count options:**
- `-j 1`: Sequential (same as no flag)
- `-j 4`: Use 4 parallel workers
- `-j 0`: Auto-detect CPU count
- `-j 16`: Use 16 workers (for large clusters)

## Best Practices

### 1. Stage Large Sweeps
```bash
# Don't run immediately - stage first to review size
yanex run exp.py --stage -p "param=range(1, 1000)"

# Check count
yanex list --status staged | wc -l

# Confirm, then run in parallel
yanex run --staged -j 8
```

### 2. Tag Staged Experiments
```bash
# Tag for easy filtering
yanex run exp.py --stage -p "..." --tag batch-2024-01-15
```

### 3. Review Before Running
```bash
# Always check what's staged
yanex list --status staged --verbose

# Look for mistakes before running
```

### 4. Use Meaningful Names
```bash
# Easier to track in logs
yanex run exp.py --stage --name "sweep-learning-rates-v2"
```

### 5. Clean Up After Running
```bash
# Verify all completed
yanex list --status staged  # Should be empty

# If any stuck, investigate
yanex show <stuck_experiment_id>
```

## Troubleshooting

### No staged experiments found
```bash
$ yanex run --staged
Error: No staged experiments found

# Check status
yanex list --status staged
```

### Experiment already running
```bash
# If experiment status is 'running', it can't be run again
yanex list --status running

# Cancel if stuck
yanex delete <experiment_id>
```

### Staged experiments from old sweep
```bash
# Clean up old staged experiments
yanex delete --status staged --before "1 week ago"
```

## Key Concepts

- **`--stage` flag**: Creates experiment without running it
- **Staged status**: Experiments wait in queue until executed
- **`yanex run --staged`**: Execute all staged experiments
- **Parallel staging**: Sweeps create multiple staged experiments at once
- **Deferred execution**: Prepare now, run later (cluster, overnight, batch)
- **Queue management**: List, filter, cancel staged experiments before running

## Next Steps

- Stage a parameter sweep and run it in parallel
- Try staging experiments on one machine, running on another
- See example 05 to learn about training loops with step-based logging

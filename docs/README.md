# Yanex Documentation

Complete documentation for Yanex - Yet Another Experiment Tracker

## Getting Started

- [**Installation & Quick Start**](../README.md#quick-start) - Get up and running in minutes
- [**Configuration**](configuration.md) - Parameter management and config files
- [**Experiment Structure**](experiment-structure.md) - Directory layout and file organization
- [**Dependencies**](dependencies.md) - Multi-stage pipelines and experiment dependencies
- [**Best Practices**](best-practices.md) - Recommended patterns and workflows
- [**AI & Automation**](ai-usage.md) - Machine-readable output, `yanex get`, and Claude Code skill

## CLI Commands

Yanex provides a comprehensive command-line interface for managing experiments.

**[→ CLI Commands Overview](cli-commands.md)** - Complete guide with common patterns

### Core Commands
- [**`yanex run`**](commands/run.md) - Execute experiments with parameter tracking
- [**`yanex list`**](commands/list.md) - List and filter experiments
- [**`yanex show`**](commands/show.md) - Display detailed experiment information
- [**`yanex compare`**](commands/compare.md) - Interactive experiment comparison

### Management Commands
- [**`yanex archive`**](commands/archive.md) - Archive old experiments
- [**`yanex unarchive`**](commands/unarchive.md) - Restore archived experiments
- [**`yanex delete`**](commands/delete.md) - Permanently delete experiments
- [**`yanex update`**](commands/update.md) - Modify experiment metadata

### Utility Commands
- [**`yanex get`**](commands/get.md) - Extract field values (AI/scripting-friendly)
- [**`yanex ui`**](commands/ui.md) - Launch web interface
- [**`yanex open`**](commands/open.md) - Open experiment directory


## Python API

Yanex provides two APIs for programmatic access:

- [**Run API**](run-api.md) - Create and execute experiments programmatically (advanced patterns)
- [**Results API**](results-api.md) - Query and analyze completed experiments
- [**Python API Overview**](python-api.md) - API usage patterns and examples

## Examples

Practical demonstrations for all Yanex features:

- [**CLI Examples**](../examples/cli/) - Main usage patterns (10 examples)
- [**Run API Examples**](../examples/run-api/) - Advanced programmatic patterns (3 examples)
- [**Results API Examples**](../examples/results-api/) - Analysis with Jupyter notebooks (2 notebooks)


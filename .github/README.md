# GitHub Actions & Templates

This directory contains GitHub Actions workflows and templates for the Yanex project.

## Workflows

### 🔄 `ci.yml` - Full CI Pipeline
**Triggers**: Push to `main`/`develop`, all pull requests
- **Lint Job**: Format checking, linting (ruff), type checking (mypy)
- **Test Job**: Matrix testing across Python 3.8-3.12 and Ubuntu/macOS/Windows
- **Build Job**: Package building and artifact upload
- **Integration Job**: CLI functionality testing

### ⚡ `feature-branch.yml` - Fast Development Feedback  
**Triggers**: Push to feature branches (not main/develop)
- **Quick Check**: Format, lint, type check, basic tests
- **Core Tests**: Python 3.9/3.11/3.12 testing with focus on recent changes
- Optimized for speed with caching and reduced matrix

### 🚀 `release.yml` - Release Automation
**Triggers**: Tags (`v*`), releases, manual dispatch
- **Validation**: Comprehensive testing with 90% coverage requirement
- **Publishing**: Automated PyPI publishing (requires `PYPI_TOKEN` secret)

## Templates

### 📝 Pull Request Template
Comprehensive checklist covering:
- Change description and type
- Testing requirements  
- CLI impact assessment
- Breaking changes documentation

### 🐛 Bug Report Template
Structured bug reporting with:
- Environment details
- Reproduction steps
- Expected vs actual behavior
- Command output capture

### ✨ Feature Request Template  
Feature planning template with:
- Problem statement
- Proposed CLI interface
- Use cases and alternatives

## Configuration

### 🤖 Dependabot
- **Python dependencies**: Weekly updates on Mondays
- **GitHub Actions**: Weekly updates on Mondays
- Automatic reviewer assignment
- Conventional commit messages

## Caching Strategy

All workflows use aggressive caching:
- **pip cache**: Based on requirements files and Python version
- **Separate caches**: Feature branches vs main/develop
- **Multi-level fallback**: Ensures cache hits across similar environments

## Coverage & Quality

- **Code Coverage**: Codecov integration with XML reports
- **Quality Gates**: Format, lint, and type checking required
- **Test Coverage**: 90% minimum for releases
- **Multi-OS Testing**: Ubuntu (primary), macOS, Windows

## Secrets Required

For full functionality, configure these repository secrets:
- `PYPI_TOKEN`: PyPI API token for automated publishing
- `CODECOV_TOKEN`: (Optional) Codecov token for private repos

## Local Development

Run the same checks locally:
```bash
make check          # All quality checks
make test           # Test suite
make test-cov       # Coverage testing
make lint           # Linting only
make format-check   # Format checking
```
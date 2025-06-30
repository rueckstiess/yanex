# Contributing to Yanex

Thank you for your interest in contributing to Yanex! We welcome contributions from the community and are excited to work with you.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/rueckstiess/yanex.git
   cd yanex
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev]"
   # Or using requirements files
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. **Verify installation**
   ```bash
   # Run tests
   pytest
   
   # Check code style
   ruff check .
   ruff format --check .
   
   # Type checking
   mypy yanex/
   ```

## Development Workflow

### Code Style

We use modern Python tooling for consistent code style:

- **Formatting**: `ruff format` (replaces black)
- **Linting**: `ruff check` (replaces flake8, isort, and more)
- **Type checking**: `mypy`

Before submitting changes:
```bash
# Format code
ruff format .

# Fix linting issues
ruff check --fix .

# Run type checking
mypy yanex/
```

### Testing

All tests must pass before merging:

```bash
# Run all tests with coverage
pytest

# Run specific test files
pytest tests/core/test_manager.py
pytest tests/cli/test_run.py

# Run tests matching a pattern
pytest -k "test_sweep"
```

**Test Guidelines:**
- Write tests for all new functionality
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use descriptive test names that explain what's being tested

### Commit Messages

We follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```
feat(sweeps): add parameter sweep functionality
fix(cli): handle empty experiment list gracefully
docs(readme): add parameter sweep examples
test(core): add tests for experiment manager
```

## Types of Contributions

### 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment information**:
   - Python version
   - Yanex version
   - Operating system
5. **Minimal code example** if applicable

### 💡 Feature Requests

For new features:

1. **Check existing issues** to avoid duplicates
2. **Describe the use case** - why is this needed?
3. **Propose a solution** - how should it work?
4. **Consider alternatives** - are there other approaches?

### 🔧 Code Contributions

1. **Start with an issue** - discuss before implementing
2. **Fork and create a branch** from `main`
3. **Make your changes** following our guidelines
4. **Add tests** for new functionality
5. **Update documentation** if needed
6. **Submit a pull request**

### 📚 Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples and use cases
- Improve API documentation
- Write tutorials or guides

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**
   ```bash
   pytest
   ruff check .
   mypy yanex/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(scope): your descriptive message"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Fill out PR template** with:
   - Description of changes
   - Related issues
   - Testing performed
   - Breaking changes (if any)

### PR Review Process

- All PRs require review before merging
- CI tests must pass
- Code coverage should not decrease
- Documentation must be updated for public APIs


## Architecture Overview

Understanding the codebase structure:

```
yanex/
├── core/          # Core experiment tracking logic
│   ├── manager.py     # Experiment lifecycle management
│   ├── storage.py     # File system storage backend
│   ├── config.py      # Configuration and parameter handling
│   └── ...
├── cli/           # Command-line interface
│   ├── commands/      # Individual CLI commands
│   └── main.py        # CLI entry point
├── api.py         # Public Python API
└── utils/         # Shared utilities
```

**Key Components:**
- **ExperimentManager**: Orchestrates experiment lifecycle
- **Storage**: Handles file operations and metadata
- **Config**: Parameter parsing and sweep expansion
- **CLI**: User-facing command interface

## Release Process

We use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Releases are managed by maintainers and follow this process:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag
4. Publish to PyPI

## Getting Help

- **Questions**: Open a GitHub discussion
- **Bugs**: Create an issue with the bug template
- **Features**: Open an issue with the feature template
- **Chat**: [Add Discord/Slack link if available]

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- Be respectful and constructive
- Welcome newcomers and help them learn
- Focus on what's best for the community
- Show empathy towards other contributors

Thank you for contributing to Yanex! 🚀
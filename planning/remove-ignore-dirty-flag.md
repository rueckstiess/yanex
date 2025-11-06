# Remove `--ignore-dirty` Flag and Add Automatic Git Patch Storage

**Status**: Planning
**Created**: 2025-11-06
**Target Version**: v0.7.0

## Overview

Remove the `--ignore-dirty` flag and its enforcement of clean git state. Instead, automatically capture and store a patch file of uncommitted changes in each experiment's artifact directory. This improves the development experience while maintaining full reproducibility.

## Motivation

The `--ignore-dirty` flag is overly restrictive during development and experiment evaluation. Users almost always run with this flag during active development, which defeats its reproducibility purpose. A better approach is to automatically capture uncommitted changes as a patch file, allowing seamless experimentation while preserving the ability to reproduce any run.

## Design Decisions

### 1. Patch File Scope

**Decision**: Capture only tracked files (staged + unstaged changes), exclude untracked files.

**Rationale**:
- **Size control**: Prevents massive patches from binary artifacts in working directory
- **Intent-based**: Files matter for reproducibility should be tracked in git
- **Escape hatch**: Users can `git add` without committing to include files
- **Performance**: Keeps patches manageable even with large untracked build artifacts
- **Git philosophy**: Respects user's implicit/explicit exclusion intent

**Implementation**: Use `git diff HEAD` to capture both staged and unstaged changes relative to HEAD.

### 2. Error Handling

**Decision**: Warn and continue if patch generation fails.

**Rationale**:
- Patch generation failure shouldn't block experiment execution
- Log clear warning for debugging
- Mark metadata with `patch_file: null` and error note
- User can still run experiments even if git operations fail

### 3. Metadata Changes

**Decision**: Keep metadata minimal with two new git fields:
```json
{
  "git": {
    "commit_hash": "...",
    "commit_hash_short": "...",
    "branch": "...",
    "author": "...",
    "message": "...",
    "committed_date": "...",
    "has_uncommitted_changes": true,
    "patch_file": "artifacts/git_diff.patch"
  }
}
```

**Fields**:
- `has_uncommitted_changes`: Boolean indicating if working directory was dirty
- `patch_file`: Relative path to patch artifact, or `null` if clean/failed

### 4. Artifact Storage

**Decision**: Store patch as `git_diff.patch` in experiment's artifacts directory.

**Path**: `~/.yanex/experiments/{exp_id}/artifacts/git_diff.patch`

**Behavior**:
- Only create file if uncommitted changes exist
- Don't create empty patch files for clean working directories
- Use `storage.save_text_artifact()` for consistent artifact handling

### 5. Config File Cleanup

**Decision**: Remove `ignore_dirty` support from config files completely.

**Rationale**: Since enforcement is removed, the config option is obsolete.

**Migration**: No migration needed (user is only user, no backwards compatibility required).

### 6. Deprecation Warning

**Decision**: Treat `--ignore-dirty` as deprecated flag with warning.

**Behavior**:
- Accept flag without error (allows old scripts to work)
- Log deprecation warning: "Warning: --ignore-dirty flag is deprecated and no longer necessary. Git patches are now captured automatically."
- Flag has no effect on execution

**Rationale**: Graceful transition for existing scripts and documentation references.

## Implementation Plan

### Phase 1: Remove Dirty State Enforcement

**Files to modify**:

#### `yanex/cli/commands/run.py`
- Keep `--ignore-dirty` option but mark as deprecated
- Add deprecation warning when flag is used
- Remove early validation calls (lines 404-411, 732-739)
- Remove `ignore_dirty` parameter propagation to manager/executor
- Remove `resolved_ignore_dirty` logic (line 219)

#### `yanex/core/manager.py`
- Remove `allow_dirty` parameter from `create_experiment()` signature
- Remove validation call `validate_clean_working_directory()` (lines 411-413)
- Update all callers within the file

#### `yanex/executor.py`
- Remove `allow_dirty` parameter from `run_multiple()` signature
- Update docstring

#### `yanex/api.py`
- Remove `allow_dirty` from `create_experiment()` if exposed
- Update any wrapper functions

**Estimated complexity**: Medium (signature changes propagate through codebase)

### Phase 2: Add Patch Generation

**Files to modify**:

#### `yanex/core/git_utils.py`

Add new functions:

```python
def has_uncommitted_changes(repo: git.Repo | None = None) -> bool:
    """Check if working directory has uncommitted changes.

    Args:
        repo: Git repository instance. If None, detects from cwd.

    Returns:
        True if uncommitted changes exist, False if clean.
    """
    # Implementation: check repo.is_dirty()


def generate_git_patch(repo: git.Repo | None = None) -> str | None:
    """Generate patch of all uncommitted changes (staged + unstaged).

    Captures differences between HEAD and working directory, including
    both staged and unstaged changes for tracked files only. Untracked
    files are excluded.

    Args:
        repo: Git repository instance. If None, detects from cwd.

    Returns:
        Patch string if changes exist, None if working directory is clean.
        Returns None (not empty string) for clean state.

    Raises:
        GitError: If git operations fail (caller should handle gracefully).
    """
    # Implementation:
    # 1. Check if any changes exist (optimize for common clean case)
    # 2. If clean, return None
    # 3. Generate: repo.git.diff('HEAD') to get both staged + unstaged
    # 4. Return patch string
```

**Key implementation notes**:
- Use `repo.git.diff('HEAD')` to capture both staged and unstaged changes
- Binary files handled automatically by git ("Binary files differ")
- Empty string vs None distinction: None = clean, empty string = edge case
- Wrap git operations in try/except to raise GitError on failure

#### `yanex/core/manager.py`

Update `create_experiment()` method:

```python
# After collecting git info (around line 480)
git_info = get_current_commit_info()

# Capture patch if uncommitted changes exist
git_patch = None
patch_filename = None
try:
    git_patch = generate_git_patch()
    if git_patch:
        patch_filename = "git_diff.patch"
except GitError as e:
    logger.warning(f"Failed to generate git patch: {e}")
    # Continue without patch

# Add to git_info
git_info["has_uncommitted_changes"] = git_patch is not None
git_info["patch_file"] = patch_filename

# Later, after metadata is created and saved (around line 520)
if git_patch:
    try:
        self.storage.save_text_artifact(
            experiment_id,
            patch_filename,
            git_patch
        )
    except Exception as e:
        logger.warning(f"Failed to save git patch artifact: {e}")
        # Don't fail experiment creation
```

**Error handling strategy**:
- Try/except around both generation and storage
- Log warnings on failure but continue
- Update metadata to reflect actual state (patch_file may be None even if changes existed)

#### `yanex/core/storage_artifacts.py`

Verify `save_text_artifact()` exists and can handle multiline strings. Add if needed:

```python
def save_text_artifact(
    self,
    experiment_id: str,
    filename: str,
    content: str
) -> None:
    """Save text content as an artifact.

    Args:
        experiment_id: Experiment ID
        filename: Artifact filename
        content: Text content to save
    """
```

**Estimated complexity**: Medium (new functions + integration into experiment lifecycle)

### Phase 3: Update Tests

**Test files to modify**:

#### `tests/core/test_git_utils.py`

**Remove**:
- `TestValidateCleanWorkingDirectory` class (lines 83-189)
  - All tests for dirty state validation no longer needed

**Add**:
- `TestHasUncommittedChanges` class:
  - Test clean working directory returns False
  - Test modified file returns True
  - Test staged file returns True
  - Test untracked file returns False (not counted as uncommitted)
  - Test multiple changes returns True

- `TestGenerateGitPatch` class:
  - Test clean working directory returns None
  - Test modified file generates patch
  - Test staged file generates patch
  - Test both staged + unstaged in same patch
  - Test untracked files excluded from patch
  - Test binary file handling (includes "Binary files differ")
  - Test patch format is valid (starts with "diff --git")
  - Test multiple file changes in single patch
  - Test git error handling (raises GitError)

#### `tests/core/test_manager.py`

**Remove**:
- Tests for `allow_dirty` parameter
- Tests for dirty state rejection

**Add**:
- Test patch generation on experiment creation:
  - Test clean working directory (no patch created)
  - Test dirty working directory (patch created)
  - Test patch stored as artifact
  - Test metadata updated with patch info
  - Test patch generation failure (warning logged, experiment continues)

#### `tests/cli/test_parameter_sweeps.py`

**Modify**:
- Remove all `--ignore-dirty` flags from test commands (9 occurrences)
- Tests should pass without flag (no enforcement)
- Optionally: Add test verifying `--ignore-dirty` shows deprecation warning

#### `tests/cli/test_direct_sweep_execution.py`

**Remove**:
- `test_direct_sweep_execution_requires_clean_git()` (lines 300-339)
  - Test is obsolete (no longer enforces clean state)

**Add**:
- Test that sweep works with dirty state
- Test that `--ignore-dirty` shows deprecation warning

#### `tests/cli/test_staged_execution.py`

**Modify**:
- Remove `--ignore-dirty` usage from tests
- Verify tests pass with dirty state

#### New file: `tests/core/test_git_patch_integration.py`

Add integration tests for complete patch workflow:
- Test end-to-end patch creation in real experiment
- Test parallel execution creates separate patches per experiment
- Test patch can be applied to reproduce state (basic validation)

**Estimated complexity**: Medium (mix of deletions and new test cases)

### Phase 4: Update Documentation

#### `docs/commands/run.md`

**Remove** (line 56):
```markdown
- `--ignore-dirty`: Allow execution with uncommitted changes
```

**Remove** (lines 320-355):
Entire "Clean State Enforcement" section including best practices.

**Add** new section after "Git Integration":

```markdown
### Automatic Patch Capture

Yanex automatically captures uncommitted changes (staged and unstaged) as a git patch file in each experiment's artifacts directory. This ensures full reproducibility even when running experiments from a dirty working directory.

**What's captured**:
- Modified tracked files (staged and unstaged)
- Deleted tracked files
- Changes are captured as a standard git diff patch

**What's excluded**:
- Untracked files (use `git add` to include files without committing)
- Binary files are noted as "Binary files differ"

**Patch location**:
```
~/.yanex/experiments/{experiment_id}/artifacts/git_diff.patch
```

**Metadata tracking**:
The experiment's `metadata.json` includes:
```json
{
  "git": {
    "commit_hash": "abc123...",
    "branch": "main",
    "has_uncommitted_changes": true,
    "patch_file": "artifacts/git_diff.patch"
  }
}
```

**Clean working directory**:
If your working directory is clean (no uncommitted changes), no patch file is created and `has_uncommitted_changes` is set to `false`.

**Reproducing experiments**:
To reproduce an experiment with uncommitted changes:
1. Check out the commit: `git checkout {commit_hash}`
2. Apply the patch: `git apply ~/.yanex/experiments/{exp_id}/artifacts/git_diff.patch`
3. Run the script with the same parameters

See the [reproduce command](#reproduce-command) for automated reproduction (coming soon).
```

**Update** (around line 35):
Replace mention of `--ignore-dirty` in examples with note that dirty state is handled automatically.

#### `README.md`

**Update** (line 35):
```markdown
- 🔒 **Reproducible**: Automatic Git state tracking and patch capture ensures every experiment is reproducible, even from dirty working directories
```

**Add** to features section:
```markdown
- Automatic git diff patch capture for uncommitted changes
```

#### `CLAUDE.md`

**Update** "Working with Experiments" section:
```markdown
- Git state is automatically tracked for reproducibility (commit hash, branch, uncommitted changes)
- Uncommitted changes are captured as patch files in experiment artifacts
- No clean state enforcement - experiments can run from dirty working directories
```

**Update** "Two Execution Patterns" section:
Remove any mentions of `--ignore-dirty` flag.

**Update** "Programmatic Batch Execution API" examples:
Remove `allow_dirty=True` from all code examples.

#### `planning/config-cli-defaults.md`

**Remove**:
All references to `ignore_dirty` configuration option.

#### `planning/programmatic-multi-experiment-execution.md`

**Remove**:
All references to `allow_dirty` parameter in API examples.

**Estimated complexity**: Medium (multiple documentation files, careful rewording)

### Phase 5: Config File Cleanup

#### Files to modify:

**Check and remove** `ignore_dirty` handling from:
- `yanex/core/config.py` - Remove default value handling
- Any config file parsing logic
- Config validation logic

**Add warning** if deprecated config found:
- When loading config, check if `ignore_dirty` key exists
- Log deprecation warning: "Config option 'ignore_dirty' is deprecated and ignored"
- Don't fail, just warn

**Test updates**:
- Remove any tests that use `ignore_dirty` in config files
- Add test for deprecation warning

**Estimated complexity**: Low (config system may not have explicit handling)

## High-Level Future Work: Reproduce Command

**Note**: Detailed planning and implementation will occur in a separate session.

### Command Overview

Add new top-level command: `yanex reproduce {experiment_id}`

**Purpose**: Re-run an experiment with the exact same parameters and git state.

**Basic usage**:
```bash
# Reproduce with current git state
yanex reproduce abc12345

# Restore git state from experiment
yanex reproduce abc12345 --restore-git

# Interactive confirmation mode
yanex reproduce abc12345 --restore-git --interactive

# Override parameters
yanex reproduce abc12345 --param learning_rate=0.001
```

### Key Features

1. **Load experiment metadata**: Script path, parameters, git info
2. **Git state restoration** (optional with `--restore-git`):
   - Check out experiment's commit hash
   - Apply patch file if exists
   - Warn about detached HEAD state
   - Require clean working directory before restoration
3. **Execute experiment**: Run script with original parameters
4. **Safety features**:
   - Dry-run mode to preview actions
   - Interactive confirmation before git operations
   - Option to restore original git state after completion

### Implementation Location

- New file: `yanex/cli/commands/reproduce.py`
- New git utilities: `git_utils.checkout_commit()`, `git_utils.apply_patch()`
- Reuse experiment execution logic from `run.py`
- Comprehensive tests for reproduction workflow

### Design Considerations

1. **Safety**: Require clean working directory before `--restore-git`
2. **Clarity**: Show clear warnings about git state changes
3. **Flexibility**: Allow parameter overrides for experimentation
4. **Automation**: Support batch reproduction of multiple experiments

### Open Questions (To Resolve Later)

- Should `--restore-git` be the default or opt-in?
- How to handle detached HEAD cleanup after reproduction?
- Should we create a new experiment ID for reproduced runs or link to original?
- How to handle reproduction when original commit is not in local repo?

## Backwards Compatibility

**Breaking changes**: Yes, but acceptable.

**Justification**:
- User is only user of yanex currently
- User frequently uses `--ignore-dirty`, so change improves workflow
- No need to preserve old experiment reproduction (all used `--ignore-dirty` anyway)

**Migration strategy**: None needed.

## Testing Strategy

### Unit Tests
- Git patch generation (clean, dirty, mixed states)
- Patch storage and metadata updates
- Error handling for failed patch operations
- Deprecated flag handling

### Integration Tests
- End-to-end experiment creation with patches
- Parallel execution with patches
- CLI flag deprecation warnings

### Manual Testing Checklist
- [ ] Run experiment with clean working directory (no patch created)
- [ ] Run experiment with modified file (patch created)
- [ ] Run experiment with staged file (patch created)
- [ ] Run experiment with untracked file (not in patch)
- [ ] Run experiment with binary file changes (handled gracefully)
- [ ] Run parameter sweep with dirty state (multiple patches created)
- [ ] Use `--ignore-dirty` flag (shows deprecation warning, still works)
- [ ] Verify metadata has new fields
- [ ] Verify patch file location and format
- [ ] Apply patch manually to verify reproducibility

## Success Criteria

1. ✅ `--ignore-dirty` flag removed or deprecated
2. ✅ No clean state enforcement during experiment creation
3. ✅ Patch files automatically created for dirty working directories
4. ✅ Patch files excluded for clean working directories
5. ✅ Metadata includes `has_uncommitted_changes` and `patch_file` fields
6. ✅ All tests pass without `--ignore-dirty` flags
7. ✅ Documentation updated to reflect new behavior
8. ✅ Error handling gracefully handles patch generation failures
9. ✅ Untracked files excluded from patches
10. ✅ Parallel execution works correctly with patches

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Patch generation failure breaks experiments | High | Warn and continue, don't fail experiment |
| Large patches from binary files | Medium | Git handles binaries gracefully with "Binary files differ" |
| Patch application fails during reproduction | Medium | Clear error messages, manual fallback instructions |
| Tests fail in CI with dirty state | Low | Tests should create clean repos via fixtures |
| Breaking existing scripts | Low | Deprecation warning, graceful handling |

## Timeline Estimate

- **Phase 1** (Remove enforcement): 2 hours
- **Phase 2** (Add patch generation): 3 hours
- **Phase 3** (Update tests): 3 hours
- **Phase 4** (Documentation): 2 hours
- **Phase 5** (Config cleanup): 1 hour

**Total**: ~11 hours for phases 1-5

**Phase 6** (Reproduce command): Separate session, estimate 8-12 hours

## References

- Git diff documentation: https://git-scm.com/docs/git-diff
- GitPython diff API: https://gitpython.readthedocs.io/en/stable/tutorial.html#obtaining-diff-information
- Git apply documentation: https://git-scm.com/docs/git-apply

## Approval

**Status**: Awaiting approval

**Reviewer**: @thomas

**Questions for reviewer**: None at this time.

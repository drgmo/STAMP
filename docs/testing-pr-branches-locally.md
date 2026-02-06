# Testing Pull Request Branches Locally

This guide walks you through the steps to **fetch, check out, and test a PR branch locally** in VS Code before merging it into `main`.
This is especially useful when an automated tool (e.g. GitHub Copilot) has created a PR with code changes that you want to verify on your machine first.

## Prerequisites

- Git installed and configured
- VS Code with the repository open
- A terminal (VS Code's integrated terminal works great: <kbd>Ctrl</kbd>+<kbd>`</kbd>)

## Step-by-Step Guide

### 1. Make sure your local repo is clean

Before switching branches, commit or stash any uncommitted work:

```bash
# Check for uncommitted changes
git status

# If you have changes you want to keep, stash them
git stash
```

### 2. Fetch all remote branches

Download the latest branch information from GitHub so your local Git knows about the PR branch:

```bash
git fetch origin
```

### 3. Check out the PR branch

Switch to the PR branch. Replace `<branch-name>` with the actual branch name from the PR (e.g. `copilot/integrate-attn-mil-multi-task`):

```bash
git checkout <branch-name>
```

For example:

```bash
git checkout copilot/integrate-attn-mil-multi-task
```

> **Tip:** You can find the exact branch name at the top of the PR page on GitHub, or by running `git branch -r` to list all remote branches.

### 4. Install dependencies (if changed)

If the PR modified `pyproject.toml` or `uv.lock`, re-sync your environment:

```bash
# For GPU systems
uv sync --extra build --extra gpu

# For CPU-only systems
uv sync --extra cpu
```

### 5. Run the tests

Verify that existing tests still pass and any new tests work:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the full test suite
python -m pytest tests/

# Or run a specific test file
python -m pytest tests/test_model.py -v
```

### 6. Try the new functionality

If the PR adds a new CLI command or feature, test it manually. For example:

```bash
# Check that STAMP still shows all subcommands
stamp --help

# Try any new subcommand added by the PR
stamp --config your_config.yaml <new-subcommand>
```

### 7. Switch back to your original branch

Once you are done testing, switch back to `main` (or whichever branch you were on):

```bash
git checkout main
```

If you stashed changes in step 1, restore them:

```bash
git stash pop
```

### 8. Merge the PR

If you are satisfied with the changes, you can merge the PR directly on GitHub by clicking **"Merge pull request"** on the PR page. Alternatively, merge locally:

```bash
git checkout main
git merge <branch-name>
git push origin main
```

## Quick Reference

| Step | Command |
|------|---------|
| Check status | `git status` |
| Stash work | `git stash` |
| Fetch branches | `git fetch origin` |
| List remote branches | `git branch -r` |
| Switch to PR branch | `git checkout <branch-name>` |
| Sync dependencies | `uv sync --extra cpu` |
| Run tests | `python -m pytest tests/ -v` |
| Go back to main | `git checkout main` |
| Restore stash | `git stash pop` |

## VS Code Tips

- Use the **Source Control** panel (<kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>G</kbd>) to see changed files.
- Click the branch name in the bottom-left status bar to quickly switch branches via the command palette.
- The **GitLens** extension provides an easy-to-use UI for comparing branches and viewing PR diffs.

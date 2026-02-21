#!/usr/bin/env bash
# fix-git-multitask-workflow.sh
#
# Fixes the situation where feature work was done directly on the main branch
# instead of a dedicated feature branch.
#
# Prerequisites: You are on master/main with uncommitted (or recently committed)
# multitask feature changes that should NOT be on main.

set -euo pipefail

# --- Step 1: Save current state ---
# Stage and commit any uncommitted changes on the main branch.
git checkout master
git add -A
git commit -m "feat: add multitask training branch support"

# --- Step 2: Isolate the feature onto its own branch ---
# Create feature/multitask pointing at the current commit (includes all feature work).
# We stay on master so we can reset it in step 4.
git branch feature/multitask

# --- Step 3: Create the dependent branch for the next feature ---
# feature/attention-export branches off feature/multitask because it depends on that code.
git branch feature/attention-export feature/multitask

# --- Step 4: Restore main to match the remote ---
# Fetch latest remote state and hard-reset master to match origin/main.
# The feature commits are preserved on feature/multitask and feature/attention-export.
git fetch origin main
git reset --hard origin/main

# --- Verification ---
echo ""
echo "=== Verification ==="
echo ""
echo "master (should match origin/main):"
git log --oneline master -3
echo ""
echo "feature/multitask (should include feature commits):"
git log --oneline feature/multitask -3
echo ""
echo "feature/attention-export (should match feature/multitask):"
git log --oneline feature/attention-export -3
